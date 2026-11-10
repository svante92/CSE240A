//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include "predictor.h"

//
// TODO:Student Information
//
const char *studentName = "Jiancheng Xiong, Svante Aretun";
const char *studentID   = "200480108, ";
const char *email       = "jix123@ucsd.edu, svante92@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = { "Static", "Gshare",
                          "Tournament", "Custom" };

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

// Gshare keeps one global history register and a table of 2-bit counters.
// The table has 2^ghistoryBits entries and is indexed by
// (low PC bits) XOR (global history).
static uint8_t *gshareBHT = NULL;
static uint32_t gshareHistory = 0;
static uint32_t gshareMask = 0;

// Index used by both prediction and training. History is still the value
// that produced the prediction, because training updates it afterward.
static uint32_t
gshare_index(uint32_t pc)
{
  return (pc ^ gshareHistory) & gshareMask;
}

static uint8_t
gshare_predict(uint32_t pc)
{
  uint8_t counter = gshareBHT[gshare_index(pc)];
  if (counter >= WT) {
    return TAKEN;
  }
  return NOTTAKEN;
}

// Saturating 2-bit update: taken moves toward ST, not-taken toward SN.
static void
gshare_train(uint32_t pc, uint8_t outcome)
{
  uint32_t index = gshare_index(pc);

  if (outcome == TAKEN) {
    if (gshareBHT[index] < ST) {
      gshareBHT[index]++;
    }
  } else if (gshareBHT[index] > SN) {
    gshareBHT[index]--;
  }

  // Shift the resolved outcome into the least significant history bit.
  gshareHistory = ((gshareHistory << 1) | outcome) & gshareMask;
}

// Tournament predictor, following the Alpha 21264 with 2-bit counters.
//
// Local predictor:
//   localHistory[2^pcIndexBits] holds lhistoryBits of per-PC history.
//   localCounters[2^lhistoryBits] is indexed by that history.
// Global predictor:
//   globalCounters[2^ghistoryBits] is indexed by the global history.
// Choice predictor:
//   choiceCounters[2^ghistoryBits] is indexed by the same global history.
//   SN and WN select the global predictor. WT and ST select the local one.
//   WN is the initial state, so the chooser weakly selects global.
static uint32_t *localHistory = NULL;
static uint8_t *localCounters = NULL;
static uint8_t *globalCounters = NULL;
static uint8_t *choiceCounters = NULL;
static uint32_t tournamentGHR = 0;
static uint32_t pcMask = 0;
static uint32_t localHistMask = 0;
static uint32_t globalMask = 0;

static uint8_t
counter_prediction(uint8_t counter)
{
  if (counter >= WT) {
    return TAKEN;
  }
  return NOTTAKEN;
}

static void
update_saturating(uint8_t *counter, uint8_t outcome)
{
  if (outcome == TAKEN) {
    if (*counter < ST) {
      (*counter)++;
    }
  } else if (*counter > SN) {
    (*counter)--;
  }
}

static void
init_counter_table(uint8_t *table, uint32_t numEntries)
{
  for (uint32_t i = 0; i < numEntries; i++) {
    table[i] = WN;
  }
}

static void
init_tournament()
{
  uint32_t numLocalHist = 1u << pcIndexBits;
  uint32_t numLocalCounters = 1u << lhistoryBits;
  uint32_t numGlobal = 1u << ghistoryBits;

  pcMask = numLocalHist - 1;
  localHistMask = numLocalCounters - 1;
  globalMask = numGlobal - 1;
  tournamentGHR = NOTTAKEN;

  // Histories start at NOTTAKEN. calloc zeroes the table.
  localHistory = calloc(numLocalHist, sizeof(uint32_t));
  localCounters = malloc(numLocalCounters * sizeof(uint8_t));
  globalCounters = malloc(numGlobal * sizeof(uint8_t));
  choiceCounters = malloc(numGlobal * sizeof(uint8_t));

  init_counter_table(localCounters, numLocalCounters);
  init_counter_table(globalCounters, numGlobal);
  init_counter_table(choiceCounters, numGlobal);
}

static uint8_t
tournament_predict(uint32_t pc)
{
  uint32_t history = localHistory[pc & pcMask] & localHistMask;
  uint8_t localPred = counter_prediction(localCounters[history]);
  uint8_t globalPred = counter_prediction(globalCounters[tournamentGHR]);

  if (choiceCounters[tournamentGHR] >= WT) {
    return localPred;
  }
  return globalPred;
}

// Train both component predictors on every branch. Move the chooser only
// when they disagree, toward whichever one was correct.
static void
tournament_train(uint32_t pc, uint8_t outcome)
{
  uint32_t pcIndex = pc & pcMask;
  uint32_t history = localHistory[pcIndex] & localHistMask;
  uint8_t localPred = counter_prediction(localCounters[history]);
  uint8_t globalPred = counter_prediction(globalCounters[tournamentGHR]);
  uint8_t *choice = &choiceCounters[tournamentGHR];

  update_saturating(&localCounters[history], outcome);
  update_saturating(&globalCounters[tournamentGHR], outcome);

  if (localPred != globalPred) {
    if (localPred == outcome) {
      if (*choice < ST) {
        (*choice)++;
      }
    } else if (*choice > SN) {
      (*choice)--;
    }
  }

  localHistory[pcIndex] = ((history << 1) | outcome) & localHistMask;
  tournamentGHR = ((tournamentGHR << 1) | outcome) & globalMask;
}


//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
void
init_predictor()
{
  uint32_t numEntries;

  switch (bpType) {
    case GSHARE:
      numEntries = 1u << ghistoryBits;
      gshareMask = numEntries - 1;
      gshareHistory = NOTTAKEN;
      gshareBHT = malloc(numEntries * sizeof(uint8_t));
      for (uint32_t i = 0; i < numEntries; i++) {
        gshareBHT[i] = WN;
      }
      break;
    case TOURNAMENT:
      init_tournament();
      break;
    default:
      break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint8_t
make_prediction(uint32_t pc)
{
  // Make a prediction based on the bpType
  switch (bpType) {
    case STATIC:
      return TAKEN;
    case GSHARE:
      return gshare_predict(pc);
    case TOURNAMENT:
      return tournament_predict(pc);
    case CUSTOM:
    default:
      break;
  }

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//
void
train_predictor(uint32_t pc, uint8_t outcome)
{
  switch (bpType) {
    case GSHARE:
      gshare_train(pc, outcome);
      break;
    case TOURNAMENT:
      tournament_train(pc, outcome);
      break;
    default:
      break;
  }
}
