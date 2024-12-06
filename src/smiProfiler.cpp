#include <iostream>
#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <map>
#include <vector>
#include "hip/hip_runtime.h"

// #include "global.h"
#include "smiProfiler.h"
#include "rocm-smi_backend.h"

#include "rocm_smi/rocm_smi.h"

#define TIME_SCRAPPER 

#define MAX_BUFFER_SIZE_MB 1024

namespace global{

  bool collect_data;
  pthread_mutex_t mutex;

}

uint64_t max_timestamps, profiler_step;
uint64_t profiler_index;  // index for each run of rocSTAR
int metrics_per_timestamp;
char *region_label;
double *metrics_buffer;
int socket_id;

std::string output_file_name = "hw_metrics.csv";
FILE *output_file;

hipEvent_t stop;
pthread_t profiler_thread;

std::vector<uint32_t> rsmi_temp_sensor_types{
  RSMI_TEMP_TYPE_JUNCTION,
  RSMI_TEMP_TYPE_MEMORY,
// RSMI_TEMP_TYPE_EDGE,
// RSMI_TEMP_TYPE_HBM_0,
// RSMI_TEMP_TYPE_HBM_1,
// RSMI_TEMP_TYPE_HBM_2,
// RSMI_TEMP_TYPE_HBM_3,
};

// Documentation here: https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/doxygen/html/rocm__smi_8h.html#adff8e3e0b004b35d90348455f807f856
std::vector<rsmi_clk_type_t> rsmi_clk_types{
  RSMI_CLK_TYPE_SYS,   // System clock
  RSMI_CLK_TYPE_SOC,   // SOC clock. 
  RSMI_CLK_TYPE_MEM,   // Memory clock.
  // RSMI_CLK_TYPE_DF,    // Data Fabric clock (for ASICs running on a separate clock)
  // RSMI_CLK_TYPE_DCEF,  // Display Controller Engine clock.
  // RSMI_CLK_TYPE_PCIE   // PCIE clock.
};   

int count_rsmi_metrics_per_timestamp(){
  int n_metrics = 2; // Timestamp and instance_index 
  n_metrics += 1;    // Power
  n_metrics += rsmi_temp_sensor_types.size();
  n_metrics += rsmi_clk_types.size();
  return n_metrics;
}


std::string get_temp_sensor_label(uint32_t sensor_type) {
  switch (sensor_type) {
    case RSMI_TEMP_TYPE_EDGE:     return "edge";
    case RSMI_TEMP_TYPE_JUNCTION: return "junction";
    case RSMI_TEMP_TYPE_MEMORY:   return "memory";
    case RSMI_TEMP_TYPE_HBM_0:    return "hbm0";
    case RSMI_TEMP_TYPE_HBM_1:    return "hbm1";
    case RSMI_TEMP_TYPE_HBM_2:    return "hbm2";
    case RSMI_TEMP_TYPE_HBM_3:    return "hbm3";
    default: return "unknown";
  }
}

std::string get_clk_label(rsmi_clk_type_t clk_type) {
  switch (clk_type) {
    case RSMI_CLK_TYPE_SYS:  return "system";
    case RSMI_CLK_TYPE_DF:   return "DF";
    case RSMI_CLK_TYPE_DCEF: return "DCE";
    case RSMI_CLK_TYPE_SOC:  return "SOC";
    case RSMI_CLK_TYPE_MEM:  return "memory";
    case RSMI_CLK_TYPE_PCIE: return "PCIE";
    default: return "unknown";
  }
}
	


void get_rocr_visible_devices() {
  const char* env_p = getenv("ROCR_VISIBLE_DEVICES");
  socket_id = 0;
  if(env_p != NULL) socket_id = atoi(env_p);
}

FILE * open_output_file(const std::string filename){
  FILE *out_file = fopen(filename.c_str(), "w");
  if (out_file == NULL) {
    perror("Error opening output file");
    exit(1);
  }
  return out_file;
}

void write_header_to_output_file( FILE *out_file ){

  fprintf( out_file, "Timestamp, Label, Instance_index" );

  // Add temperatures headers
  for (int i = 0; i < rsmi_temp_sensor_types.size(); ++i) {
    uint32_t sensor_type = rsmi_temp_sensor_types[i];
    std::stringstream temp_label;
    temp_label << ", Temp_" << get_temp_sensor_label(sensor_type) << "[C]";
    fprintf( out_file, temp_label.str().c_str() );
  }

  // Add power headers
  fprintf( out_file, ", Power[W]" );

  // Add clocks headers
  for (int i = 0; i < rsmi_clk_types.size(); ++i) {
    rsmi_clk_type_t clk_type = rsmi_clk_types[i];
    std::stringstream clk_label;
    clk_label << ", Clk_" << get_clk_label(clk_type) << "[MHz]";
    fprintf( out_file, clk_label.str().c_str() );
  }

  fprintf( out_file, " \n");
}

void smiProfilerInit(){
 
  profiler_step = 0;
  profiler_index = 0;

  metrics_per_timestamp = count_rsmi_metrics_per_timestamp();
  printf( "N metrics per timestamp: %d \n", metrics_per_timestamp );
  
  get_rocr_visible_devices();
  double MB_per_timestamp = (double)(metrics_per_timestamp * sizeof(double)) / 1024 / 1024;  // Using double type
  max_timestamps = floor( (double) MAX_BUFFER_SIZE_MB / MB_per_timestamp);  //Max number of timestamps we are going to collect
  printf( "max_timesteps: %ld \n", max_timestamps );
  printf( "allocated buffer: %ld \n", metrics_per_timestamp * max_timestamps *sizeof(double) );
  metrics_buffer = (double*)malloc(metrics_per_timestamp * max_timestamps *sizeof(double));
  
  output_file = open_output_file(output_file_name);
  write_header_to_output_file( output_file );
  
  initialize_rocm_smi();

  global::mutex = PTHREAD_MUTEX_INITIALIZER;
  global::collect_data = false;

  std::cout << "smiProfiler Initialized" << std::endl;
}

void smiProfilerFinalize(){

  free(metrics_buffer);
 	if (fclose(output_file) != 0) {
		perror("Error closing output file.");
	} 
  std::cout << "smiProfiler Finalized" << std::endl;
}


void* thread_entry(void* arg){
	int ret;	
  ret = rocm_smi_collect_metrics( metrics_buffer, profiler_step, metrics_per_timestamp, max_timestamps, profiler_index, region_label, output_file );
  return NULL;
}


// Start collecting HW telemetry data
void smiProfilerStart(char *label){
	hipEventCreate(&stop);

	region_label = label;
  profiler_step = 0;

	global::collect_data = true;
	pthread_create(&profiler_thread, NULL, thread_entry, NULL);
}

// Stop collecting HW telemetry data
void smiProfilerStop()
{
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);

	// Increment index
	profiler_index++;

	//Stopping the loop for rsmi function
	pthread_mutex_lock(&global::mutex);
	global::collect_data = false;

	pthread_mutex_unlock(&global::mutex);

	//Wait for thread to finish
	pthread_join(profiler_thread, NULL);

}

/////////////////////////////////////////////////////////////////////////////////////
// rocm-smi backend
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

rsmi_status_t rsmi_status;

void initialize_rocm_smi(){

  rsmi_status = rsmi_init(0);
  if (rsmi_status != RSMI_STATUS_SUCCESS ){
    std::cout << "ERROR: rocm-smi failed to initialize. Exiting" << std::endl;
    exit(1);
  }

  uint32_t n_devices;
  rsmi_status = rsmi_num_monitor_devices(&n_devices);
  std::cout << "rocm-smi found " << n_devices << " devices." << std::endl;

}



void rocm_smi_metrics_scrapper( int device_id, double *data_buffer, uint64_t step, int metrics_per_step, uint64_t index ){

  uint64_t offset = step * metrics_per_step; 

  // Get current time in nanoseconds
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	uint64_t timestamp_nanoseconds = (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
  data_buffer[offset++] = (double)timestamp_nanoseconds;
  
  // Instance index
  data_buffer[offset++] = index;
  
  // Get temperature
  // Documentation here: https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/doxygen/html/group__PhysQuer.html#ga40e9da04e4c0cfa17a4f38b97ebc9669
  for (int i = 0; i < rsmi_temp_sensor_types.size(); ++i) {
    int64_t rsmi_temp;
    uint32_t sensor_type = rsmi_temp_sensor_types[i];
    rsmi_status = rsmi_dev_temp_metric_get(device_id, sensor_type, RSMI_TEMP_CURRENT, &rsmi_temp);
    if( rsmi_status == RSMI_STATUS_NOT_SUPPORTED ) rsmi_temp = 0;
    data_buffer[offset++] = rsmi_temp/1e3; // Convert to Celsius 
  }

  // Get device power
  RSMI_POWER_TYPE rsmi_power_type;
  uint64_t rsmi_power;
  rsmi_status = rsmi_dev_power_get( device_id, &rsmi_power, &rsmi_power_type);
  data_buffer[offset++] = rsmi_power / 1e6; // Power in watts

  // Get device clocks
  // From here: https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/doxygen/html/group__PerfQuer.html#gad329e66e21b7cf8826e83177b441257a
  rsmi_frequencies_t rsmi_clocks;
  double clk;
  for (int i = 0; i < rsmi_clk_types.size(); ++i) {
    rsmi_clk_type_t clk_type = rsmi_clk_types[i];
    rsmi_status = rsmi_dev_gpu_clk_freq_get(device_id, clk_type, &rsmi_clocks);
    if (rsmi_status == RSMI_STATUS_SUCCESS ) clk = rsmi_clocks.frequency[rsmi_clocks.current] / 1e6; // clock in Mhz;
    else clk = 0;
    data_buffer[offset++] = clk;
  }
  
} 

void write_rocm_smi_metrics(FILE *out_file, double *data_buffer, uint64_t n_steps, int metrics_per_step, const char *region_label) {
  if (out_file == NULL) {
    perror("Error accessing output file to append metrics.  \n");
    return;
  }
  for(int i=0; i<n_steps; ++i){
    // Write the timestamp and the instance label
    fprintf(out_file, "%.0lf, %s", (double)data_buffer[i*metrics_per_step], region_label );
    // write the instance index
    fprintf(out_file, ", %d", (int)data_buffer[i*metrics_per_step+1] );
    for(int j=2; j<metrics_per_step; ++j){
      fprintf(out_file, ", %.1lf", data_buffer[i*metrics_per_step + j]);
    }
    fprintf(out_file, "\n");
  }
}

int rocm_smi_collect_metrics( double *data_buffer, uint64_t &step, 
                              int metrics_per_step, uint64_t max_steps, 
                              uint64_t index, const char *region_label, FILE *output_file ){

  #ifdef TIME_SCRAPPER
  std::chrono::high_resolution_clock::time_point time_start, time_end;                               
  clock_t start_time, end_time;
  int sample_counter = 0;
  double scrapper_time, scrapper_time_min, scrapper_time_max, scrapper_time_sum;
  scrapper_time_max = 0;
  scrapper_time_min = 1e30;
  scrapper_time_sum = 0;
  #endif

  while(step < max_steps){
    pthread_mutex_lock(&global::mutex);

    pthread_mutex_unlock(&global::mutex);
    if(!global::collect_data) break;
    
    #ifdef TIME_SCRAPPER
    // start_time = clock();
    time_start = std::chrono::high_resolution_clock::now();
    #endif
    
    rocm_smi_metrics_scrapper( socket_id, data_buffer, step, metrics_per_step, index );
    
    #ifdef TIME_SCRAPPER
    // end_time = clock();
    time_end = std::chrono::high_resolution_clock::now();
    #endif
    
    usleep(100);

    pthread_mutex_lock(&global::mutex);
    pthread_mutex_unlock(&global::mutex);
    if(!global::collect_data) break;

    #ifdef TIME_SCRAPPER
    // scrapper_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1e3;
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start);
    scrapper_time = (double) duration.count()/1e6;     
    scrapper_time_min = fmin(scrapper_time, scrapper_time_min);
    scrapper_time_max = fmax(scrapper_time, scrapper_time_max);
    scrapper_time_sum += scrapper_time;
    sample_counter += 1;
    #endif
    
    step += 1;
  }

  #ifdef TIME_SCRAPPER
  printf("ScrapPer timer. n_samples: %d   min: %.2f  max: %.2f  mean: %.2f  millisecs \n", sample_counter, scrapper_time_min, scrapper_time_max, scrapper_time_sum/sample_counter);
  #endif 

  if(step == max_steps){
    printf("Exceeded rocSTAR max buffer size. Please increase buffer limit with -DMAX_BUFFER_SIZE_MB\n");
    exit(0);
  }

  write_rocm_smi_metrics(output_file, data_buffer, step, metrics_per_step, region_label );
  return 0;
}
