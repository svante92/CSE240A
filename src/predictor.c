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


//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
void
init_predictor()
{
  if (bpType != GSHARE) {
    return;
  }

  uint32_t numEntries = 1u << ghistoryBits;
  gshareMask = numEntries - 1;
  gshareHistory = NOTTAKEN;
  gshareBHT = malloc(numEntries * sizeof(uint8_t));
  for (uint32_t i = 0; i < numEntries; i++) {
    gshareBHT[i] = WN;
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
    default:
      break;
  }
}
