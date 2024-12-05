#ifndef SMI_PROFILER_H
#define SMI_PROFILER_H

namespace global{

  bool collect_data;
  pthread_mutex_t mutex;

}


void smiProfilerInit();
void smiProfilerFinalize();

void rocStarStart(char *label);
void rocStarStop();

#endif //SMI_PROFILER_H