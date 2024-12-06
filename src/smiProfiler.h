#ifndef SMI_PROFILER_H
#define SMI_PROFILER_H

// namespace global{

//   bool collect_data;
//   pthread_mutex_t mutex;

// }


void smiProfilerInit();
void smiProfilerFinalize();

void smiProfilerStart(char *label);
void smiProfilerStop();

#endif //SMI_PROFILER_H