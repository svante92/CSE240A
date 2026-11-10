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
const char *studentID   = "200480108, 200479310";
const char *email       = "jix123@ucsd.edu, saretun@ucsd.edu";

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

// Custom predictor: TAGE (TAgged GEometric history length).
//
// We implemented TAGE as our custom predictor. A bimodal base table gives a
// default prediction. Seven tagged tables are indexed by hashes of the PC
// and global history of geometrically increasing length (4, 8, 15, 28, 54,
// 104, 200 branches). The longest-history table whose tag matches provides
// the prediction. On a misprediction, a new entry is allocated in a table
// with longer history. 2-bit "useful" counters protect entries that help,
// and they are aged periodically so stale entries can be replaced.
//
// Storage (bits):
// tageBase[4096], 2-bit counters: 4096 * 2 = 8192
// tageTables[7][512], 3-bit counter + 2-bit useful + 10-bit tag = 15 bits
// per entry: 7 * 512 * 15 = 53760
// tageGhist[201], 200 bits of history plus the bit that falls off the end,
// needed by the folded hashes: 201
// tageFoldIdx[7], 9 bits each: 7 * 9 = 63
// tageFoldTag0[7], 10 bits each: 7 * 10 = 70
// tageFoldTag1[7], 9 bits each: 7 * 9 = 63
// tagePath, path history: 16
// tageUseAlt, 4-bit signed counter: 4
// tageTick, u-aging counter (18 bits + 1 phase bit): 19
// tageLfsr, 16-bit pseudo-random generator: 16
// Total: 62404
// Budget is 64K + 256 = 65792 bits.
#define TAGE_NUM_TABLES 7
#define TAGE_BASE_BITS 12
#define TAGE_INDEX_BITS 9
#define TAGE_TAG_BITS 10
#define TAGE_HIST_LEN 201
#define TAGE_AGE_PERIOD (1u << 18)

typedef struct {
  int8_t ctr; // 3-bit signed, -4..3. ctr >= 0 predicts taken.
  uint16_t tag;
  uint8_t u;  // 2-bit useful counter
} tage_entry;

// Global history compressed (XOR-folded) down to compLen bits, updated
// incrementally so long histories can be hashed cheaply.
typedef struct {
  uint32_t comp;
  int origLen;
  int compLen;
} tage_fold;

static const int tageHistLens[TAGE_NUM_TABLES] = { 4, 8, 15, 28, 54, 104, 200 };

static uint8_t tageBase[1 << TAGE_BASE_BITS];
static tage_entry tageTables[TAGE_NUM_TABLES][1 << TAGE_INDEX_BITS];
static uint8_t tageGhist[TAGE_HIST_LEN];  // circular, newest at tageGhistPos
static int tageGhistPos = 0;
static tage_fold tageFoldIdx[TAGE_NUM_TABLES];
static tage_fold tageFoldTag0[TAGE_NUM_TABLES];
static tage_fold tageFoldTag1[TAGE_NUM_TABLES];
static uint16_t tagePath = 0;
static int tageUseAlt = 0;  // >= 0 trusts altpred for new entries
static uint32_t tageTick = 0;
static uint16_t tageLfsr = 0xACE1u;

// State computed by tage_lookup and reused by training
static uint32_t tageIdx[TAGE_NUM_TABLES];
static uint16_t tageTag[TAGE_NUM_TABLES];
static int tageProvider;  // -1 means the base table
static int tageAltTable;  // -1 means the base table
static uint8_t tageProviderPred;
static uint8_t tageAltPred;
static uint8_t tageFinalPred;

static uint8_t
tage_ghist_bit(int age)
{
  return tageGhist[(tageGhistPos + age) % TAGE_HIST_LEN];
}

static void
tage_fold_init(tage_fold *f, int origLen, int compLen)
{
  f->comp = 0;
  f->origLen = origLen;
  f->compLen = compLen;
}

// Shift in the newest history bit and remove the one that just left the
// window. Called after the new bit is pushed into tageGhist.
static void
tage_fold_update(tage_fold *f)
{
  f->comp = (f->comp << 1) | tage_ghist_bit(0);
  f->comp ^= (uint32_t)tage_ghist_bit(f->origLen) << (f->origLen % f->compLen);
  f->comp ^= f->comp >> f->compLen;
  f->comp &= (1u << f->compLen) - 1;
}

static uint16_t
tage_random()
{
  tageLfsr = (tageLfsr >> 1) ^ (-(tageLfsr & 1u) & 0xB400u);
  return tageLfsr;
}

static void
tage_ctr_update(int8_t *ctr, uint8_t outcome)
{
  if (outcome == TAKEN) {
    if (*ctr < 3) {
      (*ctr)++;
    }
  } else if (*ctr > -4) {
    (*ctr)--;
  }
}

// A freshly allocated entry that has not proven itself yet.
static int
tage_is_weak(tage_entry *e)
{
  return (e->ctr == 0 || e->ctr == -1) && e->u == 0;
}

static void
init_tage()
{
  for (uint32_t i = 0; i < (1u << TAGE_BASE_BITS); i++) {
    tageBase[i] = WN;
  }
  for (int i = 0; i < TAGE_NUM_TABLES; i++) {
    tage_fold_init(&tageFoldIdx[i], tageHistLens[i], TAGE_INDEX_BITS);
    tage_fold_init(&tageFoldTag0[i], tageHistLens[i], TAGE_TAG_BITS);
    tage_fold_init(&tageFoldTag1[i], tageHistLens[i], TAGE_TAG_BITS - 1);
  }
}

static uint8_t
tage_lookup(uint32_t pc)
{
  uint32_t indexMask = (1u << TAGE_INDEX_BITS) - 1;
  uint32_t tagMask = (1u << TAGE_TAG_BITS) - 1;

  for (int i = 0; i < TAGE_NUM_TABLES; i++) {
    tageIdx[i] = (pc ^ (pc >> (abs(TAGE_INDEX_BITS - i) + 1))
                  ^ tageFoldIdx[i].comp ^ ((uint32_t)tagePath << i)) & indexMask;
    tageTag[i] = (pc ^ tageFoldTag0[i].comp ^ (tageFoldTag1[i].comp << 1))
                 & tagMask;
  }

  // Provider is the longest matching table, alt is the next longest.
  tageProvider = -1;
  tageAltTable = -1;
  for (int i = TAGE_NUM_TABLES - 1; i >= 0; i--) {
    if (tageTables[i][tageIdx[i]].tag == tageTag[i]) {
      if (tageProvider < 0) {
        tageProvider = i;
      } else {
        tageAltTable = i;
        break;
      }
    }
  }

  uint8_t basePred = counter_prediction(tageBase[pc & ((1u << TAGE_BASE_BITS) - 1)]);
  if (tageAltTable >= 0) {
    tageAltPred = tageTables[tageAltTable][tageIdx[tageAltTable]].ctr >= 0;
  } else {
    tageAltPred = basePred;
  }

  if (tageProvider < 0) {
    tageFinalPred = basePred;
    return tageFinalPred;
  }

  tage_entry *e = &tageTables[tageProvider][tageIdx[tageProvider]];
  tageProviderPred = e->ctr >= 0;
  if (tage_is_weak(e) && tageUseAlt >= 0) {
    tageFinalPred = tageAltPred;
  } else {
    tageFinalPred = tageProviderPred;
  }
  return tageFinalPred;
}

static uint8_t
tage_predict(uint32_t pc)
{
  return tage_lookup(pc);
}

static void
tage_train(uint32_t pc, uint8_t outcome)
{
  uint8_t *base = &tageBase[pc & ((1u << TAGE_BASE_BITS) - 1)];

  tage_lookup(pc);

  // On a misprediction, allocate an entry in a longer-history table.
  // Randomly skipping one table spreads allocations out.
  if (tageFinalPred != outcome && tageProvider < TAGE_NUM_TABLES - 1) {
    int start = tageProvider + 1;
    if (start < TAGE_NUM_TABLES - 1 && (tage_random() & 1)) {
      start++;
    }
    int allocated = 0;
    for (int i = start; i < TAGE_NUM_TABLES && !allocated; i++) {
      tage_entry *e = &tageTables[i][tageIdx[i]];
      if (e->u == 0) {
        e->tag = tageTag[i];
        e->ctr = outcome == TAKEN ? 0 : -1;
        allocated = 1;
      }
    }
    // Nothing was free, so make the candidates easier to replace next time.
    if (!allocated) {
      for (int i = tageProvider + 1; i < TAGE_NUM_TABLES; i++) {
        tage_entry *e = &tageTables[i][tageIdx[i]];
        if (e->u > 0) {
          e->u--;
        }
      }
    }
  }

  if (tageProvider >= 0) {
    tage_entry *e = &tageTables[tageProvider][tageIdx[tageProvider]];

    // Learn whether new entries or the alternate prediction are more reliable.
    if (tage_is_weak(e) && tageProviderPred != tageAltPred) {
      if (tageAltPred == outcome) {
        if (tageUseAlt < 7) {
          tageUseAlt++;
        }
      } else if (tageUseAlt > -8) {
        tageUseAlt--;
      }
    }

    tage_ctr_update(&e->ctr, outcome);

    // While the provider is not yet useful, keep the alternate trained too.
    if (e->u == 0) {
      if (tageAltTable >= 0) {
        tage_ctr_update(&tageTables[tageAltTable][tageIdx[tageAltTable]].ctr,
                        outcome);
      } else {
        update_saturating(base, outcome);
      }
    }

    if (tageProviderPred != tageAltPred) {
      if (tageProviderPred == outcome) {
        if (e->u < 3) {
          e->u++;
        }
      } else if (e->u > 0) {
        e->u--;
      }
    }
  } else {
    update_saturating(base, outcome);
  }

  // Periodically age the useful counters, clearing the high and low bit
  // on alternate periods.
  tageTick++;
  if (tageTick % TAGE_AGE_PERIOD == 0) {
    uint8_t keep = (tageTick / TAGE_AGE_PERIOD) & 1 ? 1 : 2;
    for (int i = 0; i < TAGE_NUM_TABLES; i++) {
      for (uint32_t j = 0; j < (1u << TAGE_INDEX_BITS); j++) {
        tageTables[i][j].u &= keep;
      }
    }
  }

  // Push the outcome into the history, then update the folded copies.
  tageGhistPos = (tageGhistPos + TAGE_HIST_LEN - 1) % TAGE_HIST_LEN;
  tageGhist[tageGhistPos] = outcome;
  tagePath = (tagePath << 1) | ((pc >> 2) & 1);
  for (int i = 0; i < TAGE_NUM_TABLES; i++) {
    tage_fold_update(&tageFoldIdx[i]);
    tage_fold_update(&tageFoldTag0[i]);
    tage_fold_update(&tageFoldTag1[i]);
  }
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
    case CUSTOM:
      init_tage();
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
      return tage_predict(pc);
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
    case CUSTOM:
      tage_train(pc, outcome);
      break;
    default:
      break;
  }
}
