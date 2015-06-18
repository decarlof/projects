#include "logfileclass.h"

//_____________________________________________________________________________________

LogFileClass::LogFileClass (const char *log_file_path, const char *log_file_name)
{
int		loop;
char	message[32];

	path = (char *) malloc ((sizeof(char) * strlen (log_file_path)) + 1);
	strcpy (path, log_file_path);

	name = (char *) malloc ((sizeof(char) * strlen (log_file_name)) + 1);
	strcpy (name, log_file_name);

	full_file_name = (char *) malloc ((sizeof(char) * strlen(log_file_path)) + (sizeof(char) * strlen(log_file_name)) + 1);

	strcpy (full_file_name, log_file_path);
	strcat (full_file_name, log_file_name);

	log_file = fopen (full_file_name, "w");

    for (loop=0;loop<MAX_TIMERS;loop++)
    {
    	timer_created[loop] = 0;

    	accumulated_time[loop].time = 0;
    	accumulated_time[loop].millitm = 0;

    	start_time[loop].time = 0;
    	start_time[loop].millitm = 0;

    	stop_time[loop].time = 0;
    	stop_time[loop].millitm = 0;

    	timer_in_use[loop] = 0;

        sprintf (message, "Timer %d", loop);
        strcpy (timer_message[loop], message);
    }
}

//_____________________________________________________________________________________

void LogFileClass::MessageLevel (int level)
{
	report_level = level;
}

//_____________________________________________________________________________________

void LogFileClass::TimeStamp (const char *message)
{
/*
	Output a message with a timestamp of the current time.
*/
struct timeb	current_time;
struct tm		*date_time_info;
char			time_buffer[256];

	ftime (&current_time);
    date_time_info = localtime (&current_time.time);
    sprintf(time_buffer, TIME_FORMAT,
    	        1900 + date_time_info->tm_year,
        	    1 + date_time_info->tm_mon,
        	    date_time_info->tm_mday,
        	    date_time_info->tm_hour,
        	    date_time_info->tm_min,
        	    date_time_info->tm_sec
                );

	fprintf (log_file, "%s -- at %s\n", message, time_buffer);
	fflush (log_file);

}

//_____________________________________________________________________________________

void LogFileClass::Message (const char *message)
{
/*
	Output a message.
*/

	fprintf (log_file, "%s\n", message);
	fflush (log_file);

	//This is a hack because without this MPI apps don't appear to honor the fflush command ^^
//	fclose (log_file);
//	log_file = fopen (full_file_name, "a");

}

//_____________________________________________________________________________________

void LogFileClass::ErrorMessage (const char *message, const char *routine)
{
/*
	Output an error message.
*/

	fprintf (log_file, "ERROR--%s\n", message);
	fprintf (log_file, "ERROR--in routine %s\n", routine);
	fflush (log_file);

}

//_____________________________________________________________________________________

void LogFileClass::WarningMessage (const char *message, const char *routine)
{
/*
	Output a warning message.
*/

	fprintf (log_file, "WARNING--%s\n", message);
	fprintf (log_file, "WARNING--in routine %s\n", routine);
	fflush (log_file);

}

//_____________________________________________________________________________________

int LogFileClass::CreateTimer (const char *message)
{
int		loop;

	loop=0;
	while ((timer_created[loop]) && (loop < 10))
       	loop++;

    if (loop < 10)
    {
		sprintf (timer_message[loop], "<%.32s>", message);
        timer_created[loop] = 1;
	    return (loop+1);
	}
    else
	    return (-1);

}

//_____________________________________________________________________________________

void LogFileClass::DestroyTimer (int timer_number)
{
	timer_number--;
	timer_created[timer_number] = 0;
}

//_____________________________________________________________________________________

void LogFileClass::StartTimer (int timer_number)
{
/*
	Mark the current time as the start time for the requested timer.
    Also mark the timer as being in use.
*/

	timer_number--;
	ftime (&start_time[timer_number]);
	timer_in_use[timer_number] = 1;
}

//_____________________________________________________________________________________

void LogFileClass::StopTimer (int timer_number)
{
/*
	Mark the current time as the stop time for the requested timer.
	Also mark the time as no longer being in use.
*/
	timer_number--;
	ftime (&stop_time[timer_number]);
	timer_in_use[timer_number] = 0;
}

//_____________________________________________________________________________________

void LogFileClass::AccumulateTimer (int timer_number)
{
/*
	If the timer is not in use--simply add to the accumulated time the time
	between the start and stop times.

	If the timer is running, stop the timer, then do the accumulation, then
	start the timer again.
*/
int		temp_milliseconds;

	timer_number--;

	if (timer_in_use[timer_number])
		StopTimer (timer_number+1);


	accumulated_time[timer_number].time += (stop_time[timer_number].time - start_time[timer_number].time);
	temp_milliseconds = (stop_time[timer_number].millitm - start_time[timer_number].millitm);

	//Adjust for the overrun
	if (temp_milliseconds < 0)
		accumulated_time[timer_number].millitm += 1000 + temp_milliseconds;
	else
		accumulated_time[timer_number].millitm += temp_milliseconds;

	if (timer_in_use[timer_number])
		StartTimer (timer_number+1);

}
//_____________________________________________________________________________________

float LogFileClass::GetTime (int timer_number)
{
float time;

		timer_number--;

	if (timer_created[timer_number])
		time = (float) accumulated_time[timer_number].time + ((float) accumulated_time[timer_number].millitm / 1000.0);
	else
		time = 0.0;

	return (time);
}

//_____________________________________________________________________________________

void LogFileClass::ResetTimer (int timer_number)
{
/*
	Reset the timer to start timing a new event.
*/
	timer_number--;
	accumulated_time[timer_number].time = 0;
	accumulated_time[timer_number].millitm = 0;

	start_time[timer_number].time = 0;
	start_time[timer_number].millitm = 0;

	stop_time[timer_number].time = 0;
	stop_time[timer_number].millitm = 0;

	timer_in_use[timer_number] = 0;
}

//_____________________________________________________________________________________

void LogFileClass::TimerMessage (int timer_number)
{
/*
	Output the message with the accumulated time for the requested timer.
    If the timer is in use, accumulate the new time before outputting the message.
    The message used for output will be the one set in CreateTimer.
*/
char    message[256];

	timer_number--;

    if (timer_in_use[timer_number])
    	AccumulateTimer (timer_number + 1);

    sprintf (message, "%s %d.%d", timer_message[timer_number], accumulated_time[timer_number].time, accumulated_time[timer_number].millitm);
    Message (message);
}

//_____________________________________________________________________________________

LogFileClass::~LogFileClass ()
{
	fclose (log_file);

	if (path != NULL)
		free (path);

	if (name != NULL)
		free (name);

	if (full_file_name != NULL)
		free (full_file_name);
}

//_____________________________________________________________________________________
