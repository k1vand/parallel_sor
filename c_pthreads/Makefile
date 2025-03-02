CC = gcc
CFLAGS = -Wall -Wextra -O3
LDFLAGS = -lm -pthread
BUILD_DIR = build
TARGET = sor
SRCS = c_pthreads.c 

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(BUILD_DIR)/$(TARGET) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
