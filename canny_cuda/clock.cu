#include <stdio.h>
#include <unistd.h>
#include "clock.h"

static int clock_type_counts[MAX_CLK_TYPES];

// exported fields
double clock_ave[MAX_CLK_TYPES];
double clock_total[MAX_CLK_TYPES];

clock_t *clock_start(void)
{
	clock_t *t;

	t = (clock_t *) malloc(sizeof(clock_t));
	*t = clock();
	return t;
}

void clock_lap(clock_t *t, int type)
{
	clock_t cur = clock();

	if (type < 0 || type >= MAX_CLK_TYPES) {
		fprintf(stderr, "invalid lap type\n");
		_exit(-2);
	}

	clock_total[type] += (double) (cur - *t) / CLOCKS_PER_SEC;
	clock_ave[type] = clock_total[type] / ++clock_type_counts[type];
	*t = cur;
}