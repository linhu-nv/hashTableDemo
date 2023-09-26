#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)

struct KeyT{
    char data[16];
    __device__ __host__ KeyT() {}
    __device__ __host__  KeyT(int64_t v1, int64_t v2) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
        ptr[1] = v2;
    }
    __device__ __host__ bool operator == (const KeyT key) {
        //int64_t* d1 = static_cast<int64_t *>((void *)data);
        //int64_t* d2 = static_cast<int64_t *>((void *)(data + 8));
        //int64_t* _d1 = static_cast<int64_t *>((void *)k.data);
        //int64_t* _d2 = static_cast<int64_t *>((void*)(k.data + 8));
        //return d1[0] == _d1[0] && d2[0] == _d2[0];
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (d1[0] == _d1[0] && d2[0] == _d2[0]) ? true : false;
    }
    __device__ __host__ void print(int matched) {
	    int* ptr = (int*)data;
	    printf("%d %d %d %d is %d\n", ptr[0], ptr[1], ptr[2], ptr[3], matched);
	    return ;
    }
};
struct ValueT{
    char data[32];
};
#define ValueBytes 32

__device__ __host__ int myHashFunc(KeyT value, int threshold) {
    //BKDR hash
    /*uint32_t seed = 31;
    uint32_t hash = 0;
    while(value) {
        hash = hash * seed + (value&0xF);
        value >>= 4;
    }
    return (hash & 0x7FFFFFFF) % threshold;*/
    //AP hash
    unsigned int hash = 0;
    int len = sizeof(KeyT);
    char* values = static_cast<char*>(value.data);
    for (int i = 0; i < len; i++) {
        if ((i & 1) == 0) {
            hash ^= ((hash << 7) ^ (values[i]&0xF) ^ (hash >> 3));
        } else {
            hash ^= (~((hash << 11) ^ (values[i]&0xF) ^ (hash >> 5)));
        }
    }
    return (hash & 0x7FFFFFFF)%threshold;
    //return ((value & 0xff)+((value>>8) & 0xff)+((value>>16) &0xff)+((value >> 24)&0xff))%threshold;

}

struct myHashTable {
    KeyT* keys;
    ValueT* values;
    int* bCount;
    int bNum;
    int bSize;
    __device__ int64_t search_key(KeyT key) {
        int hashvalue = myHashFunc(key, bNum);
        int my_bucket_size = bCount[hashvalue];
        KeyT* list = keys + (int64_t)hashvalue*bSize;
        assert((int64_t)list % 16 == 0);
        int lane_id = threadIdx.x%32;
        int my_matched = 0;
        int any_matched = 0;
        KeyT nullKey(-1, -1);
        int64_t result = -1;

        //int warp_id = (threadIdx.x + blockIdx.x*blockDim.x)/32;
        //if (lane_id == 0)
        //    key.print(-1*hashvalue);

        for (int i = lane_id; i < (my_bucket_size+31)/32*32; i += 32) {
            KeyT myKey = i < my_bucket_size ? list[i] : nullKey;

            //myKey.print(warp_id);
            
            my_matched = (myKey == key) ? 1 : 0;
            any_matched = __ballot_sync(0xffffffff, my_matched);
            if (any_matched) {
                result = hashvalue*bSize + __ffs(any_matched); 
                break;
            }
        }
        return result;
    }
};



__global__ void build_hashtable_kernel(myHashTable ht, KeyT* all_keys, ValueT* all_values, int ele_num, int* build_failure) {
    int bucket_num = ht.bNum;
    int bucket_size = ht.bSize;
    KeyT* keys = ht.keys;
    ValueT* values = ht.values;
    int* bucket_count = ht.bCount;
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int i = thread_idx; i < ele_num; i =  i+total_threads) {
        KeyT my_key = all_keys[i];
        ValueT my_value = all_values[i];
        int hashed_value = myHashFunc(my_key, bucket_num);
        int write_off = atomicAdd(bucket_count + hashed_value, 1);
        if (write_off >= bucket_size) {
            build_failure[0] = 1;
            //printf("keyIdx is %d, hashed value is %d, now size is %d, error\n", i, hashed_value, write_off);
            break;
        }
        keys[hashed_value*bucket_size + write_off] = my_key;
        values[hashed_value*bucket_size + write_off] = my_value;
    }
    return ;
}

bool buildHashTable(myHashTable &ht, KeyT* all_keys, ValueT* all_values, int bucket_num, int bucket_size, int ele_num) {
    

    ht.bNum = bucket_num;
    ht.bSize = bucket_size;

    printf("bnum is %d, bsize is %d, ele num is %d\n", bucket_num, bucket_size, ele_num);

    int total_size = ht.bNum * ht.bSize;
    CUDA_TRY(cudaMalloc((void **)&ht.keys, sizeof(KeyT)*total_size));
    CUDA_TRY(cudaMalloc((void **)&ht.values, sizeof(ValueT)*total_size));
    CUDA_TRY(cudaMalloc((void **)&ht.bCount, sizeof(int)*bucket_num));
    CUDA_TRY(cudaMemset(ht.bCount, 0, sizeof(int)*bucket_num));
    
    int* build_failure;
    CUDA_TRY(cudaMalloc((void **)&build_failure, sizeof(int)));
    CUDA_TRY(cudaMemset(build_failure, 0, sizeof(int)));

    //build hash table kernel
    //TODO: here we use atomic operations for building hash table for simplicity.
    //If we need better performance for this process, we can use multi-split.

    cudaEvent_t start, stop;
    float esp_time_gpu;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventRecord(start, 0));

    int block_size = 256;
    int block_num = 2048;
    build_hashtable_kernel<<<block_num, block_size>>>(ht, all_keys, all_values, ele_num, build_failure);
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaEventRecord(stop, 0));
    CUDA_TRY(cudaEventSynchronize(stop));
    CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
    printf("Time for build_hashtable_kernel is: %f ms\n", esp_time_gpu);

    /*int* h_hash_count = new int[bucket_num];
    cudaMemcpy(h_hash_count, ht.bCount, sizeof(int)*bucket_num, cudaMemcpyDeviceToHost);
    for (int i = 0; i < bucket_num; i ++)
        printf("%d ", h_hash_count[i]);
    printf("\n");

    KeyT *h_keys = new KeyT[bucket_num*bucket_size];
    cudaMemcpy(h_keys, ht.keys, sizeof(KeyT)*bucket_size*bucket_num, cudaMemcpyDeviceToHost);
    printf("here is the bucket:\n");
    for (int i = 0; i < bucket_num; i ++) {
        printf("bucket %d:\n", i);
        for (int j = 0; j < h_hash_count[i]; j ++) {
            h_keys[i*bucket_size + j].print(0);
        }
    }
    printf("\n");
    delete [] h_keys;
    delete [] h_hash_count;*/


    //build success check
    int* build_flag = new int[1];
    CUDA_TRY(cudaMemcpy(build_flag, build_failure,sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaDeviceSynchronize());
    bool return_state = build_flag[0] == 0 ? true : false;
    if (build_flag[0] == 1) {
        CUDA_TRY(cudaFree(ht.keys));
        CUDA_TRY(cudaFree(ht.values));
        CUDA_TRY(cudaFree(ht.bCount));
    } else {
        printf("build hash table success\n");
    }
    delete [] build_flag;
    CUDA_TRY(cudaFree(build_failure));
    return return_state;
}

__global__ void search_hashtable_kernel(myHashTable ht, KeyT* target_keys, ValueT* result_values, int64_t target_key_size, int* matched_count) {
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x)/32;
    int lane_id = threadIdx.x%32;
    ValueT* values = ht.values;
    int matched_ele = 0;
    for (int64_t i = warp_id; i < target_key_size; i += (blockDim.x*gridDim.x)/32) {

        //KeyT myKey = target_keys[i];
        //int32_t *ptr = (int32_t*)myKey.data;
        //if (lane_id == 0 && (int64_t)myKey.data % 16 != 0)
        //    printf("warp_id %d lane id %d, i %ld, mykey %lx, %d, %d, %d, %d\n", 
        //            warp_id, lane_id, i, target_keys + i, ptr[0], ptr[1], ptr[2], ptr[3]);
        
        int64_t offset = ht.search_key(target_keys[i]);
        if (offset != -1) {
            matched_ele ++;
            //output result values
            for (int b = lane_id; b < ValueBytes; b += 32) {
                result_values[i].data[b] = values[offset].data[b];
            }
        }
    }
    if (lane_id == 0)
        matched_count[warp_id] = matched_ele;
    return ;
}

int64_t searchInHashTable(myHashTable ht, KeyT* target_keys, ValueT* result_values, int64_t target_key_size) {

    int* matched_count;
    int block_size = 256;
    int block_num = 2048;
    CUDA_TRY(cudaMalloc((void **)&matched_count, sizeof(int)*block_size*block_num/32));
    CUDA_TRY(cudaMemset(matched_count, 0, sizeof(int)*block_size*block_num/32));

    cudaEvent_t start, stop;
    float esp_time_gpu;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventRecord(start, 0));

    
    search_hashtable_kernel<<<block_num, block_size>>>(ht, target_keys, result_values, target_key_size, matched_count);

    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaEventRecord(stop, 0));
    CUDA_TRY(cudaEventSynchronize(stop));
    CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
    printf("Time for search_hashtable_kernel is: %f ms, where target key size is %ld\n", esp_time_gpu, target_key_size);

    int64_t matched_num = thrust::reduce(thrust::device, matched_count, matched_count+block_size*block_num/32);
    CUDA_TRY(cudaFree(matched_count));
    return matched_num;
}

int main(int argc, char **argv) {
    //adjustable parameters
    float avg2cacheline = 0.8;
    float avg2bsize = 0.5;
    float matches2allsearch = 0.2;

    int ele_num = 100000;
    int64_t target_key_size = 128 * 1024UL;

    int cacheline_size = 128/sizeof(KeyT);
    int avg_size = cacheline_size*avg2cacheline;
    int bucket_size = avg_size/avg2bsize;
    int bucket_num = (ele_num + avg_size - 1)/avg_size;
    printf("bucket_size %d, bucket_num %d, avg_size %d\n", bucket_size, bucket_num, avg_size);

    bucket_size = bucket_size < 20 ? 20 : bucket_size;

    //generate random key-value pairs
    KeyT* all_keys;
    ValueT* all_values;
    CUDA_TRY(cudaMalloc((void **)&all_keys, sizeof(KeyT)*ele_num));
    CUDA_TRY(cudaMalloc((void **)&all_values, sizeof(ValueT)*ele_num));


    void *k = (void *)all_keys, *v = (void *)all_values;
    thrust::sequence(thrust::device, (int*)k, (int*)k + sizeof(KeyT)/4*ele_num, 0);
    thrust::sequence(thrust::device, (int*)v, (int*)v + sizeof(ValueT)/4*ele_num, 0);
    thrust::default_random_engine g;
    thrust::shuffle(thrust::device, (int*)k, (int*)k + sizeof(KeyT)/4*ele_num, g);
    thrust::shuffle(thrust::device, (int*)v, (int*)v + sizeof(KeyT)/4*ele_num, g);

    /*int* h_all_keys = new int[sizeof(KeyT)/4*ele_num];
    cudaMemcpy(h_all_keys, (int*)all_keys, sizeof(KeyT)*ele_num, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("this is the %ld generated target keys\n", sizeof(KeyT)/4*ele_num);
    for (int i = 0; i < sizeof(KeyT)/4*ele_num; i ++) printf("%d\t",h_all_keys[i]);
    printf("\n");
    delete [] h_all_keys;*/

    myHashTable ht;

    //build hash table
    while(!buildHashTable(ht, all_keys, all_values, bucket_num, bucket_size, ele_num)) {
        bucket_size = 2*bucket_size;
        printf("Build hash table failed! The avg2bsize is %f now. Rebuilding... ...\n", avg2bsize);
    }

    //generate search keys
    printf("start generating search keys...\n");
    KeyT* target_keys;
    ValueT* result_values;
    CUDA_TRY(cudaMalloc((void **)&target_keys, sizeof(KeyT)*target_key_size));
    CUDA_TRY(cudaMalloc((void **)&result_values, sizeof(ValueT)*target_key_size));
    


    int64_t copy_from_ht_num = target_key_size*matches2allsearch;
    int64_t random_gen_num = target_key_size - copy_from_ht_num;

    void *tk = (void *)target_keys;
    thrust::sequence(thrust::device, static_cast<int*>(tk), 
                                     static_cast<int*>(tk)+sizeof(KeyT)/4*random_gen_num,
                                     sizeof(KeyT)/4*copy_from_ht_num);
    thrust::shuffle(thrust::device,  static_cast<int*>(tk), 
                                     static_cast<int*>(tk)+sizeof(KeyT)/4*random_gen_num,
                                     g);
    for (int i = 0; i < (copy_from_ht_num+ele_num-1)/ele_num; i ++) {
        int64_t off = random_gen_num + ele_num*i;
        int64_t copy_size = target_key_size - off;
        copy_size = copy_size > ele_num ? ele_num : copy_size;
	    //printf("copy info: off is %ld, copy_size is %d\n", off, copy_size);
        CUDA_TRY(cudaMemcpy(target_keys + off, all_keys, sizeof(KeyT)*copy_size, cudaMemcpyDeviceToDevice));
    }
    CUDA_TRY(cudaDeviceSynchronize());

    thrust::shuffle(thrust::device, target_keys, target_keys + target_key_size, g);
    
    /*int* h_target_keys = new int[sizeof(KeyT)/4*target_key_size];
    cudaMemcpy(h_target_keys, (int*)target_keys, sizeof(KeyT)*target_key_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    random_gen_num *= sizeof(KeyT)/4;
    copy_from_ht_num *= sizeof(KeyT)/4;
    printf("this is the %ld generated target keys\n", random_gen_num);
    for (int i = 0; i < random_gen_num; i ++) printf("%d\t",h_target_keys[i]);
    printf("\n");
    printf("this is the %ld copyed target keys\n", copy_from_ht_num);
    for (int i = 0; i < copy_from_ht_num; i ++) printf("%d\t",h_target_keys[random_gen_num+i]);
    printf("\n");
    delete [] h_target_keys;
    random_gen_num /= sizeof(KeyT)/4;
    copy_from_ht_num /= sizeof(KeyT)/4;*/

    //search in the hash table
    int64_t results = searchInHashTable(ht, target_keys, result_values, target_key_size);
    if (results == copy_from_ht_num)
        printf("quick validation for hash table PASSED!\n");
    else 
        printf("quick validation for hash table FAILED! results is %ld, should be %ld\n", results, copy_from_ht_num);


    CUDA_TRY(cudaFree(all_keys));
    CUDA_TRY(cudaFree(all_values));
    CUDA_TRY(cudaFree(target_keys));
    CUDA_TRY(cudaFree(result_values));
    CUDA_TRY(cudaFree(ht.bCount));
    CUDA_TRY(cudaFree(ht.keys));
    CUDA_TRY(cudaFree(ht.values));
    return 0;
}
