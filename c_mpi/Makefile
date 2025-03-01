all:
	mpicc -Wall c_mpi.c -o ./build/sor -lm

CC = mpicc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm
BUILD_DIR = build
TARGET = sor
SRCS = c_mpi.c 

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run:
	
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
