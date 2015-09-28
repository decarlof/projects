/* File gopt.c - Test getopt() routine */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>


int main(int argc, char *argv[])
{
	extern char *optarg;
	extern int optind,opterr;
	int c;
	char cstr[2]={0,0};
	while((c=getopt(argc,argv,"g:va:"))!=EOF)
	{
		cstr[0]=c;
		printf("c= %s, optind= %d, opterr= %d, optarg = %s \n",
			cstr,optind,opterr,optarg);
	}


	printf("c= %s, optind= %d, opterr= %d, optarg = %s \n",
		cstr,optind,opterr,optarg);
	
	exit(0);
}
