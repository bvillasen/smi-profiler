#ifndef RSMI_BACKEND_H
#define RSMI_BACKEND_H

#include <stdint.h>
#include <stdio.h>
#include <iostream>


void initialize_rocm_smi();

int rocm_smi_collect_metrics( double *metrics_buffer, uint64_t &step, 
                              int metrics_per_timestamp, uint64_t max_timestamps, 
                              uint64_t index, const char *region_label, FILE *output_file );

#endif //RSMI_BACKEND_H                              