CC = xcrun clang
CFLAGS = -framework Foundation -framework IOKit -framework Metal -O2 -Wall
OBJ = m5gpu

all: $(OBJ)

$(OBJ): m5gpu.m
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OBJ)

.PHONY: all clean
