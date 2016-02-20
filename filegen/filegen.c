#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 1MB File = 1048576 bytes
// 1GB File = 1073741824 bytes
#define MEGABYTE 1<<20
#define GIGABYTE 1<<30

int main(int argc, char const *argv[])
{
	srand(2195);
	int i;
	FILE* output = fopen("../testfile", "w");
	for (i = 0; i < GIGABYTE; ++i)
		fputc(rand()%96+32, output);
	return 0;
}