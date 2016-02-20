// Written by Andrew Carter (2008)
// Modified by Tuan Dao (2016)

#include "blowfish.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

int main(int argc, char *argv[])
{
	splashscreen();
	printf("By Tuan Dao | EECE 5640 | C/OpenMP version\n");
	// Misc counters
	int status = 0;
	size_t pos = 0;
	uint64_t hash_original, hash_encrypted, hash_decrypted;
	// Clock variables
	double start, end, runtime;
	// File variables
	size_t filesize;
	char *filepath = "../testfile";
	uint32_t *file = readfile(&filesize, filepath);
	size_t numblocks = filesize/sizeof(uint32_t); // Actually 2x numblocks
	printf("File size = %zu, number of blocks = %zu\n", filesize, numblocks/2);
	// Encryption key
	char *key = "TESTKEY";
	printf("Key = %s, length = %zu\n", key, strlen(key));
	// Create Blowfish context
	blowfish_context_t *context = malloc(sizeof(blowfish_context_t));
	if(!context) 
	{
		printf("Could not allocate enough memory!\n");
		return -1;
	}

	// Initialize key schedule
	status = blowfish_init(context, key, strlen(key));
	if (status)
	{
		printf("Error initiating key\n");
		return -1;
	} else printf("Key schedule complete!\n");

	// Hash original file
	hash_original = hash(file, numblocks);
	printf("Original hash = %llx\n", hash_original);

	#pragma omp parallel
	if (omp_get_thread_num() == 0)
		printf("OpenMP is using %d threads.\n", omp_get_num_threads());

	//__________ENCRYPTION__________
	printf("Encryption starts...\n");

	start = omp_get_wtime();

	#pragma omp parallel for
	for (pos = 0; pos < numblocks; pos+=2)
	{
		blowfish_encryptblock(context, file+pos, file+pos+1);
	}

	end = omp_get_wtime();
	runtime = end - start;

	printf("Encryption done!\n");
	printf("Time taken: %lf milliseconds\n", runtime*1e3);
	printf("Average speed: %lf MB/s\n", (double)filesize/(runtime*MEGABYTE));

	hash_encrypted = hash(file, numblocks);
	printf("Encrypted hash = %llx\n", hash_encrypted);

	//__________DECRYPTION__________
	printf("Decryption starts...\n");

	start = omp_get_wtime();

	#pragma omp parallel for
	for (pos = 0; pos < numblocks; pos+=2)
	{
		blowfish_decryptblock(context, file+pos, file+pos+1);
	}

	end = omp_get_wtime();
	runtime = end - start;

	printf("Decryption done!\n");
	printf("Time taken: %lf milliseconds\n", runtime*1e3);
	printf("Average speed: %lf MB/s\n", (double)filesize/(runtime*MEGABYTE));

	hash_decrypted = hash(file, numblocks);
	printf("Decrypted hash = %llx\n", hash_decrypted);

	// Check
	if (hash_decrypted == hash_original)
		printf("Hashes match! PASSED!\n");
	else
		printf("Hashes mismatch! FAILED!\n");

	//__________DONE__________
	blowfish_clean(context);
	free(file);
	return 0;
}