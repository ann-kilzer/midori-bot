#include <stdio.h>

// function to make the sound of a セミ every 17 loops
int main() {
    int n = 1000;

    for (int i = 1; i <= n; i ++) {
      if (i % 17 == 0) {
          printf("ミンミンミン");
      }
    }

    return 0;
}
