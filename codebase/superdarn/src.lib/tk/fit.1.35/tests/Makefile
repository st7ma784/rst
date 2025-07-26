# Makefile for profiling tests in fit.1.35

# Compiler settings
CC = gcc
CFLAGS = -Wall -O3 -fopenmp -I../../include -I..
LDFLAGS = -L.. -lfit.1.35 -lm

# Test program
TARGET = profile_fit

# Source files
SOURCES = profile_fit.c
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Build the test program
$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS) -fopenmp

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Run the profiler
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all run clean
