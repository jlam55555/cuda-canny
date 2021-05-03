#include <stdio.h>
#include <unistd.h>
#include "clock.h"

static int clock_type_counts[TYPES];
static double clock_total[TYPES];

// exported field
double clock_ave[TYPES];

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

	if (type < 0 || type >= TYPES) {
		fprintf(stderr, "invalid lap type\n");
		_exit(-2);
	}

	clock_total[type] += (double) (cur - *t) / CLOCKS_PER_SEC;
	clock_ave[type] = clock_total[type] / ++clock_type_counts[type];
	*t = cur;
}