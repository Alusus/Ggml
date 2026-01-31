# Ggml for Alusus
[[العربية]](README.ar.md)

Alusus language bindings for the [GGML](https://github.com/ggml-org/ggml) tensor library.


## Usage

```
import "Apm";
Apm.importFile("Alusus/Ggml");
use Ggml;
```


## Examples

See the `Examples/` directory:
- `example.alusus` - Basic matrix multiplication using context
- `backend_example.alusus` - Matrix multiplication using backend (Vulkan/CPU)


## Vulkan Support

To enable Vulkan backend, set the environment variable before running:
```bash
export GGML_USE_VULKAN=1
```


## API Reference

### Global Functions

| Function             | Signature                                             |
|----------------------|-------------------------------------------------------|
| `init`               | `(params: InitParams): ref[Context]`                  |
| `free`               | `(ctx: ref[Context])`                                 |
| `numaInit`           | `(numaStrategy: NumaStrategy)`                        |
| `isNuma`             | `(): Bool`                                            |
| `getTensorOverhead`  | `(): ArchWord`                                        |
| `getTypeSize`        | `(type: Type): ArchWord`                              |
| `getRowSize`         | `(type: Type, ne: Int[64]): ArchWord`                 |
| `getGraphOverhead`   | `(): ArchWord`                                        |
| `setAbortCallback`   | `(cb: ptr[func(CharsPtr)]): ptr[func(CharsPtr)]`      |
| `abort`              | `(file: CharsPtr, line: Int, fmt: CharsPtr, ...any)`  |
| `fTypeToType`        | `(ftype: FType): Type`                                |
| `statusToString`     | `(status: Status): CharsPtr`                          |
| `fp16ToFp32`         | `(v: Word[16]): Float`                                |
| `fp16ToFp32`         | `(src: ref[array[Word[16]]],`                         |
|                      | ` dst: ref[array[Float]],`                            |
|                      | ` len: Int[64])`                                      |
| `fp32ToFp16`         | `(v: Float): Word[16]`                                |
| `fp32ToFp16`         | `(src: ref[array[Float]],`                            |
|                      | ` dst: ref[array[Word[16]]],`                         |
|                      | ` len: Int[64])`                                      |
| `fp32ToBf16`         | `(v: Float): Word[16]`                                |
| `fp32ToBf16`         | `(src: ref[array[Float]],`                            |
|                      | ` dst: ref[array[Word[16]]],`                         |
|                      | ` len: Int[64])`                                      |
| `bf16ToFp32`         | `(v: Word[16]): Float`                                |
| `bf16ToFp32`         | `(src: ref[array[Word[16]]],`                         |
|                      | ` dst: ref[array[Float]],`                            |
|                      | ` len: Int[64])`                                      |
| `buildForwardExpand` | `(graph: ref[CGraph], tensor: ref[Tensor])`           |
| `buildBackwardExpand`| `(context: ref[Context],`                             |
|                      | ` graph: ref[CGraph],`                                |
|                      | ` tensor: ref[ref[Tensor]])`                          |

### Status Enum

| Constant       | Value | Description               |
|----------------|-------|---------------------------|
| `ALLOC_FAILED` | -2    | Memory allocation failed  |
| `FAILED`       | -1    | Operation failed          |
| `SUCCESS`      | 0     | Operation succeeded       |
| `ABORTED`      | 1     | Operation aborted         |

### Type Enum (Data Types)

| Constant   | Value | Description                    |
|------------|-------|--------------------------------|
| `F32`      | 0     | 32-bit float                   |
| `F16`      | 1     | 16-bit float                   |
| `Q4_0`     | 2     | 4-bit quantized (variant 0)    |
| `Q4_1`     | 3     | 4-bit quantized (variant 1)    |
| `Q5_0`     | 6     | 5-bit quantized (variant 0)    |
| `Q5_1`     | 7     | 5-bit quantized (variant 1)    |
| `Q8_0`     | 8     | 8-bit quantized (variant 0)    |
| `Q8_1`     | 9     | 8-bit quantized (variant 1)    |
| `Q2_K`     | 10    | 2-bit K-quantized              |
| `Q3_K`     | 11    | 3-bit K-quantized              |
| `Q4_K`     | 12    | 4-bit K-quantized              |
| `Q5_K`     | 13    | 5-bit K-quantized              |
| `Q6_K`     | 14    | 6-bit K-quantized              |
| `IQ2_XXS`  | 16    | IQ 2-bit extra extra small     |
| `IQ2_XS`   | 17    | IQ 2-bit extra small           |
| `IQ3_XXS`  | 18    | IQ 3-bit extra extra small     |
| `IQ1_S`    | 19    | IQ 1-bit small                 |
| `IQ4_NL`   | 20    | IQ 4-bit non-linear            |
| `IQ3_S`    | 21    | IQ 3-bit small                 |
| `IQ2_S`    | 22    | IQ 2-bit small                 |
| `IQ4_XS`   | 23    | IQ 4-bit extra small           |
| `I8`       | 24    | 8-bit integer                  |
| `I16`      | 25    | 16-bit integer                 |
| `I32`      | 26    | 32-bit integer                 |
| `I64`      | 27    | 64-bit integer                 |
| `F64`      | 28    | 64-bit float                   |
| `IQ1_M`    | 29    | IQ 1-bit medium                |
| `BF16`     | 30    | Brain float 16-bit             |

### FType Enum (Format Types)

| Constant                 | Value |
|--------------------------|-------|
| `UNKNOWN`                | -1    |
| `ALL_F32`                | 0     |
| `MOSTLY_F16`             | 1     |
| `MOSTLY_Q4_0`            | 2     |
| `MOSTLY_Q4_1`            | 3     |
| `MOSTLY_Q4_1_SOME_F16`   | 4     |
| `MOSTLY_Q8_0`            | 7     |
| `MOSTLY_Q5_0`            | 8     |
| `MOSTLY_Q5_1`            | 9     |
| `MOSTLY_Q2_K`            | 10    |
| `MOSTLY_Q3_K`            | 11    |
| `MOSTLY_Q4_K`            | 12    |
| `MOSTLY_Q5_K`            | 13    |
| `MOSTLY_Q6_K`            | 14    |
| `MOSTLY_IQ2_XXS`         | 15    |
| `MOSTLY_IQ2_XS`          | 16    |
| `MOSTLY_IQ3_XXS`         | 17    |
| `MOSTLY_IQ1_S`           | 18    |
| `MOSTLY_IQ4_NL`          | 19    |
| `MOSTLY_IQ3_S`           | 20    |
| `MOSTLY_IQ2_S`           | 21    |
| `MOSTLY_IQ4_XS`          | 22    |
| `MOSTLY_IQ1_M`           | 23    |
| `MOSTLY_BF16`            | 24    |
| `MOSTLY_MXFP4`           | 25    |

### NumaStrategy Enum

| Constant     | Value |
|--------------|-------|
| `DISABLED`   | 0     |
| `DISTRIBUTE` | 1     |
| `ISOLATE`    | 2     |
| `NUMACTL`    | 3     |
| `MIRROR`     | 4     |
| `COUNT`      | 5     |

### InitParams

| Field       | Type     |
|-------------|----------|
| `memSize`   | ArchWord |
| `memBuffer` | ptr      |
| `noAlloc`   | Bool     |

### Context

| Method                | Signature                                         |
|-----------------------|---------------------------------------------------|
| `newTensor`           | `(type: Type, nDims: Int,                         |
|                       |   ne: ref[array[Int[64]]]): ref[Tensor]`          |
| `newTensor`           | `(type: Type, ne0: Int[64]): ref[Tensor]`         |
| `newTensor`           | `(type: Type, ne0: Int[64],                       |
|                       |   ne1: Int[64]): ref[Tensor]`                     |
| `newTensor`           | `(type: Type, ne0: Int[64],                       |
|                       |   ne1: Int[64], ne2: Int[64]): ref[Tensor]`       |
| `newTensor`           | `(type: Type, ne0-3: Int[64]): ref[Tensor]`       |
| `dup`                 | `(a: ref[Tensor]): ref[Tensor]`                   |
| `dup`                 | `(a: ref[CGraph], forceGrads: Bool): ref[CGraph]` |
| `newGraph`            | `(): ref[CGraph]`                                 |
| `newGraph`            | `(size: ArchWord, grads: Bool): ref[CGraph]`      |
| `backendAllocTensors` | `(backend: ref[Backend]): ref[BackendBuffer]`     |
| `add`                 | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `sub`                 | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `mul`                 | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `div`                 | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `log`                 | `(t: ref[Tensor]): ref[Tensor]`                   |
| `sum`                 | `(a: ref[Tensor]): ref[Tensor]`                   |
| `mean`                | `(a: ref[Tensor]): ref[Tensor]`                   |
| `argmax`              | `(a: ref[Tensor]): ref[Tensor]`                   |
| `norm`                | `(a: ref[Tensor]): ref[Tensor]`                   |
| `rmsNorm`             | `(a: ref[Tensor], eps: Float): ref[Tensor]`       |
| `mulMat`              | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `scale`               | `(a: ref[Tensor], alpha: Float): ref[Tensor]`     |
| `set`                 | `(a: ref[Tensor], value: Float): ref[Tensor]`     |
| `cpy`                 | `(src: ref[Tensor],`                              |
|                       | ` dst: ref[Tensor]): ref[Tensor]`                 |
| `cont`                | `(a: ref[Tensor]): ref[Tensor]`                   |
| `reshape`             | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `reshape`             | `(a: ref[Tensor], ne0: Int[64]): ref[Tensor]`     |
| `reshape`             | `(a: ref[Tensor], ne0: Int[64],`                  |
|                       | ` ne1: Int[64]): ref[Tensor]`                     |
| `reshape`             | `(a: ref[Tensor], ne0-2: Int[64]): ref[Tensor]`   |
| `reshape`             | `(a: ref[Tensor], ne0-3: Int[64]): ref[Tensor]`   |
| `view`                | `(a: ref[Tensor], ne0: Int[64],`                  |
|                       | ` offset: ArchWord): ref[Tensor]`                 |
| `view`                | `(a: ref[Tensor], ne0-1: Int[64],`                |
|                       | ` nb1: ArchWord, offset: ArchWord): ref[Tensor]`  |
| `view`                | `(a: ref[Tensor], ne0-2: Int[64],`                |
|                       | ` nb1-2: ArchWord,`                               |
|                       | ` offset: ArchWord): ref[Tensor]`                 |
| `view`                | `(a: ref[Tensor], ne0-3: Int[64],`                |
|                       | ` nb1-3: ArchWord,`                               |
|                       | ` offset: ArchWord): ref[Tensor]`                 |
| `permute`             | `(a: ref[Tensor], axis1-4: Int): ref[Tensor]`     |
| `transpose`           | `(a: ref[Tensor]): ref[Tensor]`                   |
| `getRows`             | `(a: ref[Tensor], b: ref[Tensor]): ref[Tensor]`   |
| `setRows`             | `(a: ref[Tensor], b: ref[Tensor],`                |
|                       | ` c: ref[Tensor]): ref[Tensor]`                   |
| `diag`                | `(a: ref[Tensor]): ref[Tensor]`                   |
| `diagMaskInf`         | `(a: ref[Tensor], nPast: Int): ref[Tensor]`       |
| `diagMaskInfInplace`  | `(a: ref[Tensor], nPast: Int): ref[Tensor]`       |
| `diagMaskZero`        | `(a: ref[Tensor], nPast: Int): ref[Tensor]`       |
| `diagMaskZeroInplace` | `(a: ref[Tensor], nPast: Int): ref[Tensor]`       |
| `softMax`             | `(a: ref[Tensor]): ref[Tensor]`                   |
| `rope`                | `(a: ref[Tensor], b: ref[Tensor],`                |
|                       | ` nDims: Int, mode: Int): ref[Tensor]`            |
| `pad`                 | `(t: ref[Tensor], p0-3: Int): ref[Tensor]`        |
| `roll`                | `(t: ref[Tensor], shift0-3: Int): ref[Tensor]`    |
| `graphCompute`        | `(graph: ref[CGraph], nThreads: Int): Status`     |

### Tensor

| Property/Method | Type/Signature                                     |
|-----------------|----------------------------------------------------|
| `name`          | `CharsPtr` (get/set)                               |
| `data`          | `ptr` (get)                                        |
| `dataF32`       | `ref[array[Float]]` (get)                          |
| `nElements`     | `Int[64]` (get)                                    |
| `nRows`         | `Int[64]` (get)                                    |
| `nBytes`        | `ArchWord` (get)                                   |
| `nBytesPad`     | `ArchWord` (get)                                   |
| `setInput`      | `()`                                               |
| `setOutput`     | `()`                                               |
| `setParam`      | `()`                                               |
| `setLoss`       | `()`                                               |
| `backendSet`    | `(data: ptr, offset: ArchWord, size: ArchWord)`    |
| `backendGet`    | `(data: ptr, offset: ArchWord, size: ArchWord)`    |

### Backend

| Method                | Signature                                     |
|-----------------------|-----------------------------------------------|
| `load`                | `(path: CharsPtr): ref[Reg]` (static)         |
| `cpuLoad`             | `()` (static)                                 |
| `vkLoad`              | `()` (static)                                 |
| `cpuInit`             | `(): ref[Backend]` (static)                   |
| `vkInit`              | `(devNum: ArchWord): ref[Backend]` (static)   |
| `free`                | `(ref[Backend])` (static)                     |
| `getDefaultBufferType`| `(): ref[BackendBufferType]`                  |
| `isCpu`               | `Bool` (property)                             |
| `isVk`                | `Bool` (property)                             |
| `cpuSetNThreads`      | `(threads: Int)`                              |
| `graphCompute`        | `(graph: ref[CGraph]): Status`                |
| `graphComputeAsync`   | `(graph: ref[CGraph]): Status`                |

### CGraph

| Method    | Signature                                             |
|-----------|-------------------------------------------------------|
| `plan`    | `(nThreads: Int, threadPool: ref[ThreadPool]): CPlan` |
| `compute` | `(cplan: ref[CPlan]): Status`                         |
| `reset`   | `()`                                                  |
| `print`   | `()`                                                  |

### CPlan

| Field               | Type                  |
|---------------------|-----------------------|
| `workSize`          | ArchWord              |
| `workData`          | ref[array[Word[8]]]   |
| `nThreads`          | Int                   |
| `threadPool`        | ref[ThreadPool]       |
| `abortCallback`     | ptr[function(ptr)]    |
| `abortCallbackData` | ptr                   |

### BackendBuffer

| Method | Signature                        |
|--------|----------------------------------|
| `free` | `(ref[BackendBuffer])` (static)  |

### Gallocr

| Method        | Signature                                             |
|---------------|-------------------------------------------------------|
| `new`         | `(backendBufferType: ref[BackendBufferType]):`        |
|               | `ref[Gallocr]` (static)                               |
| `free`        | `(ref[Gallocr])` (static)                             |
| `allocGraph`  | `(graph: ref[CGraph]): Bool`                          |

### ThreadPool

| Method     | Signature                                            |
|------------|------------------------------------------------------|
| `new`      | `(params: ref[Params]): ref[ThreadPool]` (static)    |
| `free`     | `(pool: ref[ThreadPool])` (static)                   |
| `nThreads` | `Int` (property)                                     |
| `pause`    | `()`                                                 |
| `resume`   | `()`                                                 |

### ThreadPool.SchedPriority Enum

| Constant   | Value |
|------------|-------|
| `LOW`      | -1    |
| `NORMAL`   | 0     |
| `MEDIUM`   | 1     |
| `HIGH`     | 2     |
| `REALTIME` | 3     |

### ThreadPool.Params

| Field       | Type             |
|-------------|------------------|
| `cpuMask`   | array[Bool, 512] |
| `nThreads`  | Int              |
| `prio`      | Int              |
| `poll`      | Word             |
| `strictCpu` | Bool             |
| `paused`    | Bool             |

| Method       | Signature                                          |
|--------------|----------------------------------------------------|
| `init`       | `(nThreads: Int)`                                  |
| `getDefault` | `(): Params` (static)                              |
| `match`      | `(a: ref[Params], b: ref[Params]): Bool` (static)  |


## Documentation

For detailed documentation of the underlying functions, refer to the [GGML documentation](https://github.com/ggml-org/ggml).


## License

This binding follows the GGML license (MIT). See the `license` file for details.
