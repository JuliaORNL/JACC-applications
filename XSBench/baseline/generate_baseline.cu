#include "XSbench_header.cuh"

int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 20;
	int mype = 0;
	double omp_start, omp_end;
	int nprocs = 1;
	unsigned long long verification;

	// Process CLI Fields -- store in "Inputs" structure
	Inputs in = read_CLI( argc, argv );

	// Print all in fields
	printf("in.nthreads: %d\n", in.nthreads);
	printf("in.n_isotopes: %ld\n", in.n_isotopes);
	printf("in.n_gridpoints: %ld\n", in.n_gridpoints);
	printf("in.lookups: %d\n", in.lookups);
	printf("in.HM: %s\n", in.HM);
	printf("in.grid_type: %d\n", in.grid_type);
	printf("in.hash_bins: %d\n", in.hash_bins);
	printf("in.particles: %d\n", in.particles);
	printf("in.simulation_method: %d\n", in.simulation_method);
	printf("in.binary_mode: %d\n", in.binary_mode);
	printf("in.kernel_id: %d\n", in.kernel_id);



	// Print-out of Input Summary
	if( mype == 0 )
		print_inputs( in, nprocs, version );

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// This is not reflective of a real Monte Carlo simulation workload,
	// therefore, do not profile this region!
	// =====================================================================
	
	SimulationData SD;

	// If read from file mode is selected, skip initialization and load
	// all simulation data structures from file instead
	if( in.binary_mode == READ )
		SD = binary_read(in);
	else
		SD = grid_init_do_not_profile( in, mype );

	// If writing from file mode is selected, write all simulation data
	// structures to file
	if( in.binary_mode == WRITE && mype == 0 )
		binary_write(in, SD);

	FILE *fp;
	// Write to file all the data of SimulationData
	fp = fopen("SimulationData.txt", "w");
	for(int i = 0; i < SD.length_num_nucs; i++)
		fprintf(fp, "%d\n", SD.num_nucs[i]);

	for(int i = 0; i < SD.length_concs; i++)
		fprintf(fp, "%a\n", SD.concs[i]);

	for(int i = 0; i < SD.length_mats; i++)
		fprintf(fp, "%d\n", SD.mats[i]);

	for(int i = 0; i < SD.length_unionized_energy_array; i++)
		fprintf(fp, "%a\n", SD.unionized_energy_array[i]);

	for(int i = 0; i < SD.length_index_grid; i++)
		fprintf(fp, "%d\n", SD.index_grid[i]);

	for(int i = 0; i < SD.length_nuclide_grid; i++)
	{
		fprintf(fp, "%a\n", SD.nuclide_grid[i].energy);
		fprintf(fp, "%a\n", SD.nuclide_grid[i].total_xs);
		fprintf(fp, "%a\n", SD.nuclide_grid[i].elastic_xs);
		fprintf(fp, "%a\n", SD.nuclide_grid[i].absorbtion_xs);
		fprintf(fp, "%a\n", SD.nuclide_grid[i].fission_xs);
		fprintf(fp, "%a\n", SD.nuclide_grid[i].nu_fission_xs);
	}
	for(int i = 0; i < SD.length_p_energy_samples; i++)
		fprintf(fp, "%a\n", SD.p_energy_samples[i]);

	for(int i = 0; i < SD.length_mat_samples; i++)
		fprintf(fp, "%d\n", SD.mat_samples[i]);
	fclose(fp);



	// Move data to GPU
	SimulationData GSD = move_simulation_data_to_device( in, mype, SD );

	printf("SD verification length: %d\n", SD.length_verification);
	printf("GSD verification length: %d\n", GSD.length_verification);

	// FILE *fp;
	// fp = fopen("Verification.txt", "w");
	// for(int i = 0; i < SD.length_verification; i++)
	// 	fprintf(fp, "%lu\n", GSD.verification[i]);

	// print all data inside num_nucs
	// for(int i = 0; i < SD.length_num_nucs; i++)
	// 	printf("num_nucs[%d]: %d\n", i, SD.num_nucs[i]);

	printf("length_num_nucs: %d\n", SD.length_num_nucs);
	printf("length_concs: %d\n", SD.length_concs);
	printf("length_mats: %d\n", SD.length_mats);
	printf("length_unionized_energy_array: %d\n", SD.length_unionized_energy_array);
	printf("length_index_grid: %ld\n", SD.length_index_grid);
	printf("length_nuclide_grid: %d\n", SD.length_nuclide_grid);
	printf("length_p_energy_samples: %d\n", SD.length_p_energy_samples);
	printf("length_mat_samples: %d\n", SD.length_mat_samples);
	printf("max_num_nucs: %d\n", SD.max_num_nucs);


	// Write to file all data inside of SimulationData
	// FILE *fp;
	// fp = fopen("SimulationLength.txt", "w");
	// fprintf(fp, "length_num_nucs: %d\n", SD.length_num_nucs);
	// fprintf(fp, "length_concs: %d\n", SD.length_concs);
	// fprintf(fp, "length_mats: %d\n", SD.length_mats);
	// fprintf(fp, "length_unionized_energy_array: %d\n", SD.length_unionized_energy_array);
	// fprintf(fp, "length_index_grid: %ld\n", SD.length_index_grid);
	// fprintf(fp, "length_nuclide_grid: %d\n", SD.length_nuclide_grid);
	// fprintf(fp, "length_p_energy_samples: %d\n", SD.length_p_energy_samples);
	// fprintf(fp, "length_mat_samples: %d\n", SD.length_mat_samples);
	// fprintf(fp, "max_num_nucs: %d\n", SD.max_num_nucs);
	// fclose(fp);

	
	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation
	// This is the section that should be profiled, as it reflects a 
	// realistic continuous energy Monte Carlo macroscopic cross section
	// lookup kernel.
	// =====================================================================
	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	// Start Simulation Timer
	omp_start = get_time();

	// Run simulation
	if( in.simulation_method == EVENT_BASED )
	{
		if( in.kernel_id == 0 )
			verification = run_event_based_simulation_baseline(in, GSD, mype);
		else if( in.kernel_id == 1 )
			verification = run_event_based_simulation_optimization_1(in, GSD, mype);
		else if( in.kernel_id == 2 )
			verification = run_event_based_simulation_optimization_2(in, GSD, mype);
		else if( in.kernel_id == 3 )
			verification = run_event_based_simulation_optimization_3(in, GSD, mype);
		else if( in.kernel_id == 4 )
			verification = run_event_based_simulation_optimization_4(in, GSD, mype);
		else if( in.kernel_id == 5 )
			verification = run_event_based_simulation_optimization_5(in, GSD, mype);
		else if( in.kernel_id == 6 )
			verification = run_event_based_simulation_optimization_6(in, GSD, mype);
		else
		{
			printf("Error: No kernel ID %d found!\n", in.kernel_id);
			exit(1);
		}
	}
	else
	{
		printf("History-based simulation not implemented in CUDA code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
		exit(1);
	}

	if( mype == 0)	
	{	
		printf("\n" );
		printf("Simulation complete.\n" );
	}

	// End Simulation Timer
	omp_end = get_time();

	// Release device memory
	release_device_memory(GSD);

	// Final Hash Step
	verification = verification % 999983;

	

	// Print / Save Results and Exit
	int is_invalid_result = print_results( in, mype, omp_end-omp_start, nprocs, verification );

	return is_invalid_result;
}
