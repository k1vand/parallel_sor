CC = gcc
CFLAGS = -O3 
LDFLAGS = -lm -fopenmp
BUILD_DIR = build
TARGET = sor
SRCS = c_omp.c 

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)

$(TARGET): 
	$(CC) $(CFLAGS) $(SRCS) -o $(BUILD_DIR)/$(TARGET) $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
