// very simple timer: can keep track of multiple clocks and multiple types;
// clock should be freed after use
#ifndef CLOCK_H
#define CLOCK_H

// maximum number of different things to benchmark (here arbitrarily set to 5)
#define MAX_CLK_TYPES 10

#include <time.h>

extern double clock_ave[MAX_CLK_TYPES];
extern double clock_total[MAX_CLK_TYPES];

clock_t *clock_start(void);
void clock_lap(clock_t *t, int type);

#endif // CLOCK_H