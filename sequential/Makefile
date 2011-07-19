CFLAGS=-g

BSEARCH_PATH=/home/rl6sf/src/binary_search/lib
MT_PATH=/home/rl6sf/src/utils/c/mt
TIMER_PATH=/home/rl6sf/src/utils/c/timer

INCLUDES=-I../lib \
		 -I$(BSEARCH_PATH)\
		 -I$(MT_PATH)\
		 -I$(TIMER_PATH)

EXECS=count_interval_intersections_seq

LIB_OBJS=../lib/interval.o \
		 $(BSEARCH_PATH)/bsearch.o \
		 $(MT_PATH)/mt.o \
		 $(TIMER_PATH)/timer.o 

all: $(EXECS)

count_interval_intersections_seq: count_interval_intersections_seq.o $(LIB_OBJS)
	gcc -o count_interval_intersections_seq \
		count_interval_intersections_seq.o \
		$(LIB_OBJS) \
		-lm

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)

