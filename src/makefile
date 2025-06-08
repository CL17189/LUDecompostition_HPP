# Makefile for LU decomposition program

CC = gcc
CFLAGS = -Wall -Werror -Wextra -O3 -fopenmp -g 
TARGET = lu
LDLIBS = -fopenmp

all: $(TARGET)

$(TARGET): lu.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET) lu.o

.PHONY: all clean

