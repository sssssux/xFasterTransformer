// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include "greedy_search.h"
#include "messenger.h"
#include "sequence.h"
#include "search_utils.h"
#include "thread_util.h"

using namespace xft;

GreedySearch::GreedySearch(AbstractDecoder &dec, const SearcherConfig &config)
    : decoder(dec), maxLen(config.maxLen), step(0), repetitionPenalty(config.repetitionPenalty) {
    eosTokenId = config.eosTokenId == -1 ? decoder.getEndId() : config.eosTokenId;
    padTokenId = config.padTokenId == -1 ? eosTokenId : config.padTokenId;
    if (repetitionPenalty <= 0) {
        printf("`repetitionPenalty` has to be a strictly positive float, but is %f.\n", repetitionPenalty);
        exit(-1);
    }
    stopWordsList = {};
    stopWordsIndex = {};
}

void GreedySearch::handleTask(int predictor_world_rank) {
    ThreadPool::getInstance().addTask([predictor_world_rank, this] {
        DecoderContext *ctx = decoder.getContext();
        if (predictor_world_rank != (ctx->ppSize - 1) * ctx->tpSize + ctx->tpRank){
            while (true) {
                int32_t sequenceID;
    #ifdef DEBUG
                printf("tsRank: %d, ppRank: %d, tpRank: %d, predictor_world_rank %d, receiving sequenceID \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, predictor_world_rank);
    #endif
                MPI_Recv(&sequenceID, 1, MPI_INT32_T, predictor_world_rank, predictor_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    #ifdef DEBUG 
                printf("tsRank: %d, ppRank: %d, tpRank: %d, predictor_world_rank %d, sequenceID received %d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, predictor_world_rank, sequenceID);
    #endif
                TimeLine t("GreedySearch.Seq" + std::to_string(sequenceID) + ".MPI_Recv");
                MPI_Recv(this->nextTokens.data(), this->batchSize, MPI_INT32_T, predictor_world_rank, predictor_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    #ifdef DEBUG 
                printf("tsRank: %d, ppRank: %d, tpRank: %d, predictor_world_rank %d, nextTokens received %d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, predictor_world_rank, this->batchSize);
    #endif
                if (SequencePool::getInstance().has(sequenceID)) {
                    auto seq = SequencePool::getInstance().get(sequenceID);
                    TaskWaitingQueue::getInstance().push(seq);
                } else {
                    printf("Error: should have sequenceID\n");
                    fflush(stdout);
                }
            }
        }else{
            int32_t sequenceID;
#ifdef DEBUG
            printf("tsRank: %d, ppRank: %d, tpRank: %d, predictor_world_rank %d, receiving sequenceID \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, predictor_world_rank);
#endif
            MPI_Recv(&sequenceID, 1, MPI_INT32_T, predictor_world_rank, predictor_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG 
            printf("tsRank: %d, ppRank: %d, tpRank: %d, predictor_world_rank %d, sequenceID received %d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, predictor_world_rank, sequenceID);
#endif
            TimeLine t("GreedySearch.Seq" + std::to_string(sequenceID) + ".MPI_Recv");
            MPI_Recv(this->nextTokens.data(), this->batchSize, MPI_INT32_T, predictor_world_rank, predictor_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG 
            printf("tsRank: %d, ppRank: %d, tpRank: %d, predictor_world_rank %d, nextTokens received %d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, predictor_world_rank, this->batchSize);
#endif
            if (SequencePool::getInstance().has(sequenceID)) {
                auto seq = SequencePool::getInstance().get(sequenceID);
                TaskWaitingQueue::getInstance().push(seq);
            } else {
                printf("Error: should have sequenceID\n");
                fflush(stdout);
            }
        }
    });
}
std::vector<int> GreedySearch::syncToken(std::tuple<float *, int, int> &result) {
    // send data from last predictor stage to first embedding stage in pipeline parallel
#if defined(PIPELINE_PARALLEL) || defined(TOKEN_SPLIT_INFER)
    DecoderContext *ctx = decoder.getContext();
    // Messenger &messenger = decoder.getMessenger();
#ifdef DEBUG
    printf("tsRank: %d, ppRank: %d, tpRank: %d, syncToken \n", ctx->tsRank, ctx->ppRank, ctx->tpRank);
#endif
    if (std::get<0>(result) == nullptr) { // The first embedding pipeline parallel stage
#ifdef DEBUG
        printf("tsRank: %d, ppRank: %d, tpRank: %d, result nullptr, enabledBackgroundSync = %d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, enabledBackgroundSync);
#endif
        this->nextTokens = std::vector<int>(batchSize, 0);
        if (ctx->ppSize > 1 && ctx->ppRank == 0 && enabledBackgroundSync == false && (ctx->tsSize == 1 or ctx->tsRank == 1)) {
            enabledBackgroundSync = true;
            int predictor_world_rank;
            // receive nextToken from 1st token last pp rank
#ifdef DEBUG
            printf("tsRank: %d, ppRank: %d, tpRank: %d, setting predictor_world_rank \n", ctx->tsRank, ctx->ppRank, ctx->tpRank);
#endif
            if (ctx->tsSize > 1){
                printf("1st handleTask will be called now.\n");
                predictor_world_rank = 0 * (ctx->tpSize * ctx->ppSize) + (ctx->ppSize - 1) * ctx->tpSize + ctx->tpRank;
                handleTask(predictor_world_rank); 
            }
            // receive nextToken from 2nd+ token last pp rank
            predictor_world_rank = ctx->tsRank * (ctx->tpSize * ctx->ppSize) + (ctx->ppSize - 1) * ctx->tpSize + ctx->tpRank;
            printf("Second handleTask will be called now.\n");
            handleTask(predictor_world_rank);   
        }
    } else { // The last predictor pipeline parallel stage
        this->nextTokens = this->search(result);
        if (ctx->ppSize > 1 && ctx->ppRank == ctx->ppSize - 1) {
            TimeLine t("GreedySearch.Seq" + std::to_string(ctx->sequenceID) + ".MPI_Send");
            int embedding_world_rank = 0;
            int predictor_world_rank = 0;
            if (ctx->tsSize >1 && ctx->tsRank == 0){
                embedding_world_rank = 1 * (ctx->tpSize * ctx->ppSize) + 0 * ctx->tpSize + ctx->tpRank;
                predictor_world_rank = 0 * (ctx->tpSize * ctx->ppSize) + (ctx->ppSize - 1) * ctx->tpSize + ctx->tpRank;
            }else{
                embedding_world_rank = ctx->tsRank * (ctx->tpSize * ctx->ppSize) + 0 * ctx->tpSize + ctx->tpRank;
                predictor_world_rank = ctx->tsRank * (ctx->tpSize * ctx->ppSize) + (ctx->ppSize - 1) * ctx->tpSize + ctx->tpRank;
            }
            
#ifdef DEBUG 
            printf("tsRank: %d, ppRank: %d, tpRank: %d, embedding_world_rank %d,  predictor_world_rank %d, sending sequenceID \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, embedding_world_rank, predictor_world_rank);
#endif
            MPI_Send(&ctx->sequenceID, 1, MPI_INT32_T, embedding_world_rank, predictor_world_rank, MPI_COMM_WORLD);
#ifdef DEBUG 
            printf("tsRank: %d, ppRank: %d, tpRank: %d, embedding_world_rank %d,  predictor_world_rank %d, sequenceID sent %d\n", ctx->tsRank, ctx->ppRank, ctx->tpRank, embedding_world_rank, predictor_world_rank, ctx->sequenceID);
#endif
            MPI_Send(this->nextTokens.data(), batchSize, MPI_INT32_T, embedding_world_rank, predictor_world_rank,
                    MPI_COMM_WORLD);
#ifdef DEBUG 
            printf("tsRank: %d, ppRank: %d, tpRank: %d, embedding_world_rank %d, predictor_world_rank %d, nextTokens sent %d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, embedding_world_rank, predictor_world_rank, batchSize);
#endif
            // TODO: Error: different scope when dynamic loading so file
            // messenger.worldSendINT32(this->nextTokens.data(), batchSize, embedding_world_rank, predictor_world_rank);
        }
    }
#else
    this->nextTokens = this->search(result);
#endif

    this->curLen++;
    for (int batchId = 0; batchId < batchSize; ++batchId) {
        output.insert(output.begin() + (batchId + 1) * curLen - 1, nextTokens[batchId]);
    }

    return this->nextTokens;
}

// Get next tokens accoring to the prompt IDs for first token
std::vector<int> GreedySearch::getNextToken(int *ids, int batchSize, int seqLen) {
    TimeLine t("1st Token");
    this->step = 0;
    this->batchSize = batchSize;
    this->curLen = seqLen;
    this->doneBatch = std::vector<int>(batchSize, 0);

    if (!this->stopWordsList.empty()) {
        stopWordsIndex = std::vector<std::vector<int>>(stopWordsList.size(), std::vector<int>(batchSize, 0));
    }

    this->output.resize(batchSize * seqLen);
    std::copy(ids, ids + batchSize * seqLen, output.begin());

    int64_t dims[3] = {batchSize, 1, seqLen};

#ifdef DEBUG
    DecoderContext *ctx = decoder.getContext();
    printf("tsRank: %d, ppRank: %d, tpRank: %d, gen 1st token \n", ctx->tsRank, ctx->ppRank, ctx->tpRank);
    printf("tsRank: %d, ppRank: %d, tpRank: %d, getNextToken dims[%ld,%ld,%ld]  \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, dims[0], dims[1], dims[2]);
#endif
    std::tuple<float *, int, int> result = decoder.forward(ids, dims, this->step++);

    return this->syncToken(result);
}

// Get next tokens according to previous predicted ID for next tokens
std::vector<int> GreedySearch::getNextToken() {
    TimeLine t("Next Token");
    int64_t dims[3] = {batchSize, 1, 1};

    DecoderContext *ctx = decoder.getContext();
#ifdef DEBUG
    printf("tsRank: %d, ppRank: %d, tpRank: %d, 2nd getNextToken dims[%ld,%ld,%ld]  \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, dims[0], dims[1], dims[2]);
#endif

    std::tuple<float *, int, int> result = decoder.forward(nextTokens.data(), dims, this->step++);

    return this->syncToken(result);
}

bool GreedySearch::isDone() {
    if (step == 0) {
        return false;
    } else if (curLen >= maxLen) {
        return true;
    } else {
        for (auto flag : doneBatch) {
            if (flag <= 0) { return false; }
        }
    }
    return true;
}

std::vector<int32_t> GreedySearch::finalize() {
    TimeLine t("dumpFile");
    t.dumpFile("timeline.json");
    return output;
}

bool GreedySearch::setStopWords(std::vector<std::vector<int>> stopWordsList) {
    this->stopWordsList = stopWordsList;
    for (auto it = this->stopWordsList.rbegin(); it != this->stopWordsList.rend(); ++it) {
        if ((*it).size() == 1 && (*it)[0] == this->eosTokenId) { this->stopWordsList.erase(std::next(it).base()); }
    }
    return !this->stopWordsList.empty();
}

std::vector<int> GreedySearch::search(std::tuple<float *, int, int> &result) {
    TimeLine t("GreedySearch");
    DecoderContext *ctx = decoder.getContext();
#ifdef DEBUG
    printf("tsRank: %d, ppRank: %d, tpRank: %d, search  \n", ctx->tsRank, ctx->ppRank, ctx->tpRank);
#endif
    float *outBuf = std::get<0>(result);
    int sampleOffset = std::get<1>(result);
    int sampleSize = std::get<2>(result);
#ifdef DEBUG
    printf("tsRank: %d, ppRank: %d, tpRank: %d, getting messenger  \n", ctx->tsRank, ctx->ppRank, ctx->tpRank);
#endif
    Messenger &messenger = decoder.getMessenger();
#ifdef DEBUG
    printf("tsRank: %d, ppRank: %d, tpRank: %d, messenger rank%d size%d color%d section%d \n", ctx->tsRank, ctx->ppRank, ctx->tpRank, messenger.getRank(), messenger.getSize(), messenger.getColor(), messenger.getSection());
#endif
    auto msgerSize = messenger.getSize();

    // Repetition penalty logits processor
    if (this->repetitionPenalty != 1.0) {
        TimeLine t("GreedySearch.repetitionPenalty");
        // step has already been incremented by 1
        if (this->step == 1) {
            this->cachedRepetVec.clear();
            this->cachedRepetVec.resize(batchSize, std::vector<int>());

            repetitionPenaltyLogitsProcess(this->repetitionPenalty, outBuf, sampleOffset, sampleSize, this->output,
                    batchSize, this->cachedRepetVec, this->step, msgerSize > 1);
        } else {
            repetitionPenaltyLogitsProcess(this->repetitionPenalty, outBuf, sampleOffset, sampleSize, this->nextTokens,
                    batchSize, this->cachedRepetVec, this->step, msgerSize > 1);
        }
    }

    // Max ID and value for each sample
    int maxIds[batchSize];
    float maxVals[batchSize];

    // Small batch size (each sample can have at least 2 threads)
    if (ctx->numThreads / batchSize >= 2) {
        int thrPerSample = ctx->numThreads / batchSize;
        int sizePerThr = (sampleSize + thrPerSample - 1) / thrPerSample;
        int maxIndices[batchSize * thrPerSample];
        float maxValues[batchSize * thrPerSample];

        // TODO: if sampleSize is small, possible to cause out of boundary
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int t = 0; t < thrPerSample; ++t) { // thread index inside the sample
                int start = t * sizePerThr;
                int end = (start + sizePerThr) > sampleSize ? sampleSize : (start + sizePerThr);
                float *p = outBuf + b * sampleSize;

                int maxIdx = start;
                float maxVal = p[start];
                for (int off = start + 1; off < end; ++off) {
                    if (p[off] > maxVal) {
                        maxVal = p[off];
                        maxIdx = off;
                    }
                }

                // False sharing happens, but since only one time, not avoided
                maxIndices[b * thrPerSample + t] = maxIdx;
                maxValues[b * thrPerSample + t] = maxVal;
            }
        }

        // Local reduction
        for (int i = 0; i < batchSize; ++i) {
            int *pIndices = maxIndices + i * thrPerSample;
            float *pValues = maxValues + i * thrPerSample;
            int maxIdx = pIndices[0];
            float maxVal = pValues[0];
            for (int j = 1; j < thrPerSample; ++j) {
                if (pValues[j] > maxVal) {
                    maxVal = pValues[j];
                    maxIdx = pIndices[j];
                }
            }
            maxIds[i] = maxIdx;
            maxVals[i] = maxVal;
        }
    }

    // Each thread handle one sample (one row)
    else {
#pragma omp parallel for
        for (int i = 0; i < batchSize; ++i) {
            int maxId = 0;
            float *p = outBuf + i * sampleSize;
            float maxVal = p[0];
            for (int j = 1; j < sampleSize; ++j) {
                if (p[j] > maxVal) {
                    maxVal = p[j];
                    maxId = j;
                }
            }
            maxIds[i] = maxId;
            maxVals[i] = maxVal;
        }
    }

    // Reduce to get the max index (any better method??)
    if (msgerSize > 1) {
        float sendBuf[2 * batchSize];
        float recvBuf[2 * batchSize * msgerSize];

        for (int i = 0; i < batchSize; ++i) {
            sendBuf[2 * i] = (float)(maxIds[i] + sampleOffset);
            sendBuf[2 * i + 1] = maxVals[i];
        }

        std::vector<long unsigned int> recvCount(msgerSize, static_cast<long unsigned int>(2 * batchSize));
        messenger.allgatherv(sendBuf, 2 * batchSize, recvBuf, recvCount);

        for (int i = 0; i < batchSize; ++i) {
            int maxId = (int)(recvBuf[2 * i] + 0.5f);
            float maxVal = recvBuf[2 * i + 1];
            for (int j = 1; j < msgerSize; ++j) {
                if (recvBuf[2 * j * batchSize + 2 * i + 1] > maxVal) {
                    maxVal = recvBuf[2 * j * batchSize + 2 * i + 1];
                    maxId = (int)(recvBuf[2 * j * batchSize + 2 * i] + 0.5f);
                }
            }
            maxIds[i] = maxId;
        }
    }

    if (eosTokenId != -1) {
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            if (doneBatch[batchId] == 0) {
                if (maxIds[batchId] == eosTokenId) { doneBatch[batchId] = 1; }
            } else if (doneBatch[batchId] > 0) {
                // Padding finished seq with padTokenId;
                maxIds[batchId] = padTokenId;
            } else if (doneBatch[batchId] < 0) {
                // Set to eosTokenId as really done;
                maxIds[batchId] = eosTokenId;
                doneBatch[batchId] = 1;
            }
        }
    }

    auto nextTokenIds_ = std::vector<int>(maxIds, maxIds + batchSize);
    if (!this->stopWordsList.empty() && !this->stopWordsIndex.empty()) {
        stopWordsCheck(nextTokenIds_, this->stopWordsList, this->stopWordsIndex, this->doneBatch);
    }

    return nextTokenIds_;
}