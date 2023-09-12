## Exercise 1 - Your first CUDA program and GPU performance metrics
1. Explain how the program is compiled and run.

    I have a make file in the code repo (I used nvcc)

2. For a vector length of N:

    1. How many floating operations are being performed in your vector add kernel? 
        
        When adding a two vectors of length N there are N plus floating point operations that are performed.

    2. How many global memory reads are being performed by your kernel? 
        
        Since both vectors are read once for each addition there is a total of 2N globalreads from global memory.

3. For a vector length of 1024:
    1. Explain how many CUDA threads and thread blocks you used. 

        I used (1024+32-1)/32 = 32 thread blocks.
        and I used 32*32=1024 CUDA threads.
        

    2. Profile your program with Nvidia Nsight. What Achieved Occupancy did you get? You might find https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-metric-comparisonLinks. useful.

        I got Achieved Occupancy of 3.12% and a Theoretical Occupancy of 50%.

        When incresing the threads per block from 32 to 64 the Theoretical Occupancy increesed to 100% and the Achieved Occupancy to 6.19%


4. Now increase the vector length to 131070:

    1. Did your program still work? If not, what changes did you make?

        The program still works. No changes needed to be made.

    2. Explain how many CUDA threads and thread blocks you used.

        I used (131070+32-1)/32 = 4096 thread blocks.
        and I used 4096*32 = 131072 CUDA threads.

    3. Profile your program with Nvidia Nsight. What Achieved Occupancy do you get now?

        Achieved Occupancy is now 32.57% (at TPB at 32) and 74.35% (at TPB at 64)


5. Further increase the vector length (try 6-10 different vector length), plot a stacked bar chart showing the breakdown of time including (1) data copy from host to device (2) the CUDA kernel (3) data copy from device to host. For this, you will need to add simple CPU timers to your code regions.

