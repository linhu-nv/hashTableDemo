#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define BINARY_SEARCH

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)

//#define KEYBYTE16
#define KEYBYTE8

#ifdef KEYBYTE16
struct KeyT{
    char data[16];
    __device__ __host__ KeyT() {}
    __device__ __host__  KeyT(int64_t v1) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
        ptr[1] = v1;
    }
    __device__ __host__ bool operator == (const KeyT key) {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (d1[0] == _d1[0] && d2[0] == _d2[0]) ? true : false;
    }
    __device__ __host__ bool operator < (const KeyT key) const {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (_d1[0] < d1[0]) ||  (_d1[0] == d1[0] && _d2[0] < d2[0]);
    }
    __device__ __host__ void print(int matched) {
	    int* ptr = (int*)data;
	    printf("%d %d %d %d is %d\n", ptr[0], ptr[1], ptr[2], ptr[3], matched);
	    return ;
    }
};
#define get_my_mask(x) 0xff<<(x/8*8)
#endif
#ifdef KEYBYTE8
struct KeyT{
    char data[8];
    __device__ __host__ KeyT() {}
    __device__ __host__  KeyT(int64_t v1) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
    }
    __device__ __host__ bool operator == (const KeyT key) {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* _d1 = (int64_t *)data;
        return d1[0] == _d1[0];
    }
    __device__ __host__ bool operator < (const KeyT key) const {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* _d1 = (int64_t *)data;
        return _d1[0] < d1[0];
    }
    __device__ __host__ void print(int matched) {
	    int* ptr = (int*)data;
	    printf("%d %d is %d\n", ptr[0], ptr[1], matched);
	    return ;
    }
};
#define get_my_mask(x) 0xf<<(x/4*4)
#endif
struct ValueT{
    char data[32];
};

#define ValueBytes 32
#define cg_size (sizeof(KeyT)/2)
//#define cg_size 8//128/16
//#define get_my_mask(x) 0xffff<<(x/16*16)


__device__ __host__ int myHashFunc(KeyT value, int threshold) {
    //BKDR hash
    uint32_t seed = 31;
    char* values = static_cast<char*>(value.data);
    int len = sizeof(KeyT);
    uint32_t hash = 171;
    while(len--) {
        char v = (~values[len-1])*(len&1) + (values[len-1])*(~(len&1));
        hash = hash * seed + (v&0xF);
    }
    return (hash & 0x7FFFFFFF) % threshold;
    //AP hash
    /*unsigned int hash = 0;
    int len = sizeof(KeyT);
    char* values = static_cast<char*>(value.data);
    for (int i = 0; i < len; i++) {
        if ((i & 1) == 0) {
            hash ^= ((hash << 7) ^ (values[i]&0xF) ^ (hash >> 3));
        } else {
            hash ^= (~((hash << 11) ^ (values[i]&0xF) ^ (hash >> 5)));
        }
    }
    return (hash & 0x7FFFFFFF)%threshold;*/
    //return ((value & 0xff)+((value>>8) & 0xff)+((value>>16) &0xff)+((value >> 24)&0xff))%threshold;

}

struct myHashTable {
    KeyT* keys;
    ValueT* values;
    int* bCount;
    int bNum;
    int bSize;
    __inline__ __device__ int64_t search_key(KeyT key, int index) {
        int hashvalue = myHashFunc(key, bNum);
        int my_bucket_size = bCount[hashvalue];
        KeyT* list = keys + (int64_t)hashvalue*bSize;

        int my_matched = 0;
        int any_matched = 0;
        KeyT nullKey(-1);
        int64_t result = -1;

        int lane_id = threadIdx.x%32;

        int aligned_size = (my_bucket_size+cg_size-1)/cg_size*cg_size;
        for (int i = threadIdx.x%cg_size; i < aligned_size; i += cg_size) {
            KeyT myKey = i < my_bucket_size ? list[i] : nullKey;
            my_matched = (myKey == key) ? 1 : 0;
            //NOTE: it is reversal! 31 30 29 28 27 26 ... 0
            any_matched = __ballot_sync(__activemask(), my_matched) & get_my_mask(lane_id);
            if (any_matched) {
                result = hashvalue*bSize + i/cg_size*cg_size + __ffs(any_matched)%cg_size; 
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
    delete [] h_hash_count;*/

    /*KeyT *h_keys = new KeyT[bucket_num*bucket_size];
    cudaMemcpy(h_keys, ht.keys, sizeof(KeyT)*bucket_size*bucket_num, cudaMemcpyDeviceToHost);
    printf("here is the bucket:\n");
    for (int i = 0; i < bucket_num; i ++) {
        printf("bucket %d:\n", i);
        for (int j = 0; j < h_hash_count[i]; j ++) {
            h_keys[i*bucket_size + j].print(0);
        }
    }
    printf("\n");
    delete [] h_keys;*/
    


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
    int tile_id = (threadIdx.x + blockIdx.x * blockDim.x)/cg_size;
    int tile_lane_id = threadIdx.x%cg_size;
    ValueT* values = ht.values;
    int matched_ele = 0;
    for (int64_t i = tile_id; i < target_key_size; i += (blockDim.x*gridDim.x)/cg_size) {
        int64_t offset = ht.search_key(target_keys[i], i);
        if (offset != -1) {
            matched_ele ++;
            //output result values
            for (int b = tile_lane_id; b < ValueBytes; b += cg_size) {
                result_values[i].data[b] = values[offset].data[b];
            }
        }
    }
    //NOTE: this change with the cg_size!
    if (cg_size <= 4)   matched_ele += __shfl_down_sync(0xffffffff, matched_ele, 4);
    if (cg_size <= 8)   matched_ele += __shfl_down_sync(0xffffffff, matched_ele, 8);
    matched_ele += __shfl_down_sync(0xffffffff, matched_ele, 16);
    if (threadIdx.x%32 == 0)
        matched_count[tile_id*cg_size/32] = matched_ele;
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

__global__ void binary_search_kernel(KeyT* bs_list, ValueT* allValues, int ele_num, 
                                     KeyT* target_keys, ValueT* result_values, int64_t target_key_size,
                                     int* match_count) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int result = 0;
    for (int64_t i = tid; i < target_key_size; i += blockDim.x*gridDim.x) {
        KeyT my_key = target_keys[i];
        int s = 0, e = ele_num, mid;
        while (s < e) {
            mid = (s+e)/2;
            if (my_key < bs_list[mid]) {
                e = mid;
            } else if (my_key == bs_list[mid]){
                result ++;
                for (int b = 0; b < ValueBytes; b ++) 
                    result_values[i].data[b] = allValues[mid].data[b];
                break;
            } else {
                s = mid + 1;
            }
        }
    }
    result += __shfl_down_sync(0xffffffff, result, 16);
    result += __shfl_down_sync(0xffffffff, result, 8);
    result += __shfl_down_sync(0xffffffff, result, 4);
    result += __shfl_down_sync(0xffffffff, result, 2);
    result += __shfl_down_sync(0xffffffff, result, 1);
    if (threadIdx.x%32 == 0) {
        match_count[tid/32] = result;
    }
    return ;
}


int64_t binarySearch(KeyT* bs_list, ValueT* all_values, int ele_num, 
                     KeyT* target_keys, ValueT* result_values, int64_t target_key_size){
    int block_size = 256;
    int block_num = 2048;
    int* match_count;
    CUDA_TRY(cudaMalloc((void **)&match_count, sizeof(int)*block_size/32*block_num));
    CUDA_TRY(cudaMemset(match_count, 0, sizeof(int)*block_size/32*block_num));

    cudaEvent_t start, stop;
    float esp_time_gpu;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventRecord(start, 0));

    binary_search_kernel<<<block_num, block_size>>>(bs_list, all_values, ele_num, 
                                                    target_keys, result_values, target_key_size, match_count);
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaEventRecord(stop, 0));
    CUDA_TRY(cudaEventSynchronize(stop));
    CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
    printf("Time for binary_search_kernel is: %f ms, where target key size is %ld\n", esp_time_gpu, target_key_size);

    int64_t result = thrust::reduce(thrust::device, match_count, match_count + block_size*block_num/32);
    CUDA_TRY(cudaFree(match_count));
    return result;
}

int main(int argc, char **argv) {
    //adjustable parameters
    float avg2cacheline = 0.7;
    float avg2bsize = 0.55;
    float matches2allsearch = 0.2;

    int ele_num = 100000;
    int64_t target_key_size =  10 *1024UL * 1024UL;

    int cacheline_size = 128/sizeof(KeyT);
    int avg_size = cacheline_size*avg2cacheline;
    int bucket_size = avg_size/avg2bsize;
    int bucket_num = (ele_num + avg_size - 1)/avg_size;
    printf("bucket_size %d, bucket_num %d, avg_size %d\n", bucket_size, bucket_num, avg_size);

    //bucket_size = bucket_size < 20 ? 20 : bucket_size;

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

    //********print the list to be searched************
    /*int* h_all_keys = new int[sizeof(KeyT)/4*ele_num];
    cudaMemcpy(h_all_keys, (int*)all_keys, sizeof(KeyT)*ele_num, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("this is the %ld generated target keys\n", sizeof(KeyT)/4*ele_num);
    for (int i = 0; i < sizeof(KeyT)/4*ele_num; i ++) printf("%d\t",h_all_keys[i]);
    printf("\n");
    delete [] h_all_keys;*/

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
    
    //*******print the target_keys**********
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

#ifdef BINARY_SEARCH
    KeyT* bs_list;
    CUDA_TRY(cudaMalloc((void **)&bs_list, sizeof(KeyT)*ele_num));
    CUDA_TRY(cudaMemcpy(bs_list, all_keys, sizeof(KeyT)*ele_num, cudaMemcpyDeviceToDevice));
    CUDA_TRY(cudaDeviceSynchronize());
    thrust::sort(thrust::device, bs_list, bs_list+ele_num);
    int64_t bs_result = binarySearch(bs_list, all_values, ele_num, target_keys, result_values, target_key_size);
    if (bs_result == copy_from_ht_num)
        printf("quick validation for binary search PASSED!\n\n");
    else 
        printf("quick validation for binary search FAILED! results is %ld, should be %ld\n\n", bs_result, copy_from_ht_num);
#endif

    myHashTable ht;

    //build hash table
    while(!buildHashTable(ht, all_keys, all_values, bucket_num, bucket_size, ele_num)) {
        bucket_size = 1.4*bucket_size;
        printf("Build hash table failed! The avg2bsize is %f now. Rebuilding... ...\n", avg2bsize);
    }

    //search in the hash table
    int64_t results = searchInHashTable(ht, target_keys, result_values, target_key_size);
    if (results == copy_from_ht_num)
        printf("quick validation for hash table PASSED!\n");
    else 
        printf("quick validation for hash table FAILED! results is %ld, should be %ld\n", results, copy_from_ht_num);

#ifdef BINARY_SEARCH
    CUDA_TRY(cudaFree(bs_list));
#endif
    CUDA_TRY(cudaFree(all_keys));
    CUDA_TRY(cudaFree(all_values));
    CUDA_TRY(cudaFree(target_keys));
    CUDA_TRY(cudaFree(result_values));
    CUDA_TRY(cudaFree(ht.bCount));
    CUDA_TRY(cudaFree(ht.keys));
    CUDA_TRY(cudaFree(ht.values));
    return 0;
}
