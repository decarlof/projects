#ifndef LOGFILECLASS_H
#define LOGFILECLASS_H

//_____________________________________________________________________________________

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/timeb.h>
#include <time.h>

//_____________________________________________________________________________________
#define	TIME_FORMAT			"%04d-%02d-%02d %02d:%02d:%02d"
//_____________________________________________________________________________________
#define	MAX_TIMERS			10
//_____________________________________________________________________________________

class LogFileClass
{
public:
	LogFileClass (const char *log_file_path, const char *log_file_name);

	void MessageLevel (int level);

	void TimeStamp (const char *message);

	void Message (const char *message);
	void ErrorMessage (const char *message, const char *routine);
	void WarningMessage (const char *message, const char *routine);

    int CreateTimer (const char *message);
	void DestroyTimer (int timer_number);
    void StartTimer (int timer_number);
    void StopTimer (int timer_number);
	void AccumulateTimer (int timer_number);
	float GetTime (int timer_number);
    void ResetTimer (int timer_number);
    void TimerMessage (int timer_number);

	~LogFileClass ();

private:
FILE		*log_file;
char		*path,
			*name,
			*full_file_name,
            timer_message[MAX_TIMERS][50];
int			report_level;
timeb		accumulated_time[MAX_TIMERS],
			start_time[MAX_TIMERS],
            stop_time[MAX_TIMERS];
int			timer_created[MAX_TIMERS],
			timer_in_use[MAX_TIMERS];

};

//_____________________________________________________________________________________
#endif
