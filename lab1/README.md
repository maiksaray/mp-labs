This folder contains implementation of parallel matrix multiplication using OpemMP

Overall approach to solution is to parallel (1) or (2): 
```     
        (1)  for (int l = 0; l < n * t; l++) {
                  int i = l % n;
                  int j = l / n;
     
        (2)       for (int k = 0; k < m; k++) {
                      c[i][j] = c[i][j] + a[i][k] * b[k][j];
                  }
              }
```  
This solution uses jinja2 python template engine to gereate c++ sources with various params in (1) or (2) and checks what works better
