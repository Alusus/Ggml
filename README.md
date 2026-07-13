# Ggml for Alusus

[[العربية]](README.ar.md)

Alusus language bindings for the [GGML](https://github.com/ggml-org/ggml) tensor library.

## Usage

```
import "Apm";
Apm.importPackage("Alusus/Ggml@0.2");
use Ggml;
```

## Examples

See the `Examples/` directory:

* `example.alusus` - Basic matrix multiplication using context
* `backend_example.alusus` - Matrix multiplication using backend (Vulkan/CPU)

## Vulkan Support

To enable Vulkan backend, pass the `ggml_enable_vulkan` option to Alusus compiler when running/compiling
your app:

```
alusus --opt ggml_enable_vulkan my_app.alusus
```

## API Reference

### Global Functions

#### getBuildDependencies

```
func getBuildDependencies(): Array[String];
```

A function that return an array of libraries and packages required to build a binary version of the
application.

#### init

```
func init (params: InitParams): ref[Context]
```

#### free

```
func free (ctx: ref[Context])
```

#### numaInit

```
func numaInit (numaStrategy: NumaStrategy)
```

#### isNuma

```
func isNuma (): Bool
```

#### getTensorOverhead

```
func getTensorOverhead (): ArchWord
```

#### getTypeSize

```
func getTypeSize (type: Type): ArchWord
```

#### getRowSize

```
func getRowSize (type: Type, ne: Int[64]): ArchWord
```

#### getGraphOverhead

```
func getGraphOverhead (): ArchWord
```

#### setAbortCallback

```
func setAbortCallback (cb: ptr[func(CharsPtr)]): ptr[func(CharsPtr)]
```

#### abort

```
func abort (file: CharsPtr, line: Int, fmt: CharsPtr, ...any)
```

#### fTypeToType

```
func fTypeToType (ftype: FType): Type
```

#### statusToString

```
func statusToString (status: Status): CharsPtr
```

#### fp16ToFp32

```
func fp16ToFp32 (v: Word[16]): Float
func fp16ToFp32 (src: ref[array[Word[16]]], dst: ref[array[Float]], len: Int[64])
```

#### fp32ToFp16

```
func fp32ToFp16 (v: Float): Word[16]
func fp32ToFp16 (src: ref[array[Float]], dst: ref[array[Word[16]]], len: Int[64])
```

#### fp32ToBf16

```
func fp32ToBf16 (v: Float): Word[16]
func fp32ToBf16 (src: ref[array[Float]], dst: ref[array[Word[16]]], len: Int[64])
```

#### bf16ToFp32

```
func bf16ToFp32 (v: Word[16]): Float
func bf16ToFp32 (src: ref[array[Word[16]]], dst: ref[array[Float]], len: Int[64])
```

#### buildForwardExpand

```
func buildForwardExpand (graph: ref[CGraph], tensor: ref[Tensor])
```

#### buildBackwardExpand

```
func buildBackwardExpand (context: ref[Context], graph: ref[CGraph], tensor: ref[ref[Tensor]])
```

### Status Enum

* `ALLOC_FAILED` (-2): Memory allocation failed
* `FAILED` (-1): Operation failed
* `SUCCESS` (0): Operation succeeded
* `ABORTED` (1): Operation aborted

### Type Enum

* `F32` (0): 32-bit float
* `F16` (1): 16-bit float
* `Q4_0` (2): 4-bit quantized (variant 0)
* `Q4_1` (3): 4-bit quantized (variant 1)
* `Q5_0` (6): 5-bit quantized (variant 0)
* `Q5_1` (7): 5-bit quantized (variant 1)
* `Q8_0` (8): 8-bit quantized (variant 0)
* `Q8_1` (9): 8-bit quantized (variant 1)
* `Q2_K` (10): 2-bit K-quantized
* `Q3_K` (11): 3-bit K-quantized
* `Q4_K` (12): 4-bit K-quantized
* `Q5_K` (13): 5-bit K-quantized
* `Q6_K` (14): 6-bit K-quantized
* `IQ2_XXS` (16): IQ 2-bit extra extra small
* `IQ2_XS` (17): IQ 2-bit extra small
* `IQ3_XXS` (18): IQ 3-bit extra extra small
* `IQ1_S` (19): IQ 1-bit small
* `IQ4_NL` (20): IQ 4-bit non-linear
* `IQ3_S` (21): IQ 3-bit small
* `IQ2_S` (22): IQ 2-bit small
* `IQ4_XS` (23): IQ 4-bit extra small
* `I8` (24): 8-bit integer
* `I16` (25): 16-bit integer
* `I32` (26): 32-bit integer
* `I64` (27): 64-bit integer
* `F64` (28): 64-bit float
* `IQ1_M` (29): IQ 1-bit medium
* `BF16` (30): Brain float 16-bit

### FType Enum

* `UNKNOWN` (-1)
* `ALL_F32` (0)
* `MOSTLY_F16` (1)
* `MOSTLY_Q4_0` (2)
* `MOSTLY_Q4_1` (3)
* `MOSTLY_Q4_1_SOME_F16` (4)
* `MOSTLY_Q8_0` (7)
* `MOSTLY_Q5_0` (8)
* `MOSTLY_Q5_1` (9)
* `MOSTLY_Q2_K` (10)
* `MOSTLY_Q3_K` (11)
* `MOSTLY_Q4_K` (12)
* `MOSTLY_Q5_K` (13)
* `MOSTLY_Q6_K` (14)
* `MOSTLY_IQ2_XXS` (15)
* `MOSTLY_IQ2_XS` (16)
* `MOSTLY_IQ3_XXS` (17)
* `MOSTLY_IQ1_S` (18)
* `MOSTLY_IQ4_NL` (19)
* `MOSTLY_IQ3_S` (20)
* `MOSTLY_IQ2_S` (21)
* `MOSTLY_IQ4_XS` (22)
* `MOSTLY_IQ1_M` (23)
* `MOSTLY_BF16` (24)
* `MOSTLY_MXFP4` (25)

### NumaStrategy Enum

* `DISABLED` (0)
* `DISTRIBUTE` (1)
* `ISOLATE` (2)
* `NUMACTL` (3)
* `MIRROR` (4)
* `COUNT` (5)

### InitParams
class InitParams {
    def memSize: ArchWord;
    def memBuffer: ptr;
    def noAlloc: Bool;
}

#### memSize

```
def memSize: ArchWord;
```

#### memBuffer

```
def memBuffer: ptr;
```

#### noAlloc

```
def noAlloc: Bool;
```

### Context

#### newTensor

```
func newTensor (type: Type, nDims: Int, ne: ref[array[Int[64]]]): ref[Tensor]
func newTensor (type: Type, ne0: Int[64]): ref[Tensor]
func newTensor (type: Type, ne0: Int[64], ne1: Int[64]): ref[Tensor]
func newTensor (type: Type, ne0: Int[64], ne1: Int[64], ne2: Int[64]): ref[Tensor]
func newTensor (type: Type, ne0-3: Int[64]): ref[Tensor]
```

#### dup

```
func dup (a: ref[Tensor]): ref[Tensor]
func dup (a: ref[CGraph], forceGrads: Bool): ref[CGraph]
```

#### newGraph

```
func newGraph (): ref[CGraph]
func newGraph (size: ArchWord, grads: Bool): ref[CGraph]
```

#### backendAllocTensors

```
func backendAllocTensors (backend: ref[Backend]): ref[BackendBuffer]
```

#### add

```
func add (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
```

#### sub

```
func sub (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
```

#### mul

```
func mul (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
```

#### div

```
func div (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
```

#### log

```
func log (t: ref[Tensor]): ref[Tensor]
```

#### sum

```
func sum (a: ref[Tensor]): ref[Tensor]
```

#### mean

```
func mean (a: ref[Tensor]): ref[Tensor]
```

#### argmax

```
func argmax (a: ref[Tensor]): ref[Tensor]
```

#### norm

```
func norm (a: ref[Tensor]): ref[Tensor]
```

#### rmsNorm

```
func rmsNorm (a: ref[Tensor], eps: Float): ref[Tensor]
```

#### mulMat

```
func mulMat (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
```

#### scale

```
func scale (a: ref[Tensor], alpha: Float): ref[Tensor]
```

#### set

```
func set (a: ref[Tensor], value: Float): ref[Tensor]
```

#### cpy

```
func cpy (src: ref[Tensor], dst: ref[Tensor]): ref[Tensor]
```

#### cont

```
func cont (a: ref[Tensor]): ref[Tensor]
```

#### reshape

```
func reshape (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
func reshape (a: ref[Tensor], ne0: Int[64]): ref[Tensor]
func reshape (a: ref[Tensor], ne0-1: Int[64]): ref[Tensor]
func reshape (a: ref[Tensor], ne0-2: Int[64]): ref[Tensor]
func reshape (a: ref[Tensor], ne0-3: Int[64]): ref[Tensor]
```

#### view

```
func view (a: ref[Tensor], ne0: Int[64], offset: ArchWord): ref[Tensor]
func view (a: ref[Tensor], ne0-1: Int[64], nb1: ArchWord, offset: ArchWord): ref[Tensor]
func view (a: ref[Tensor], ne0-2: Int[64], nb1-2: ArchWord, offset: ArchWord): ref[Tensor]
func view (a: ref[Tensor], ne0-3: Int[64], nb1-3: ArchWord, offset: ArchWord): ref[Tensor]
```

#### permute

```
func permute (a: ref[Tensor], axis1-4: Int): ref[Tensor]
```

#### transpose

```
func transpose (a: ref[Tensor]): ref[Tensor]
```

#### getRows

```
func getRows (a: ref[Tensor], b: ref[Tensor]): ref[Tensor]
```

#### setRows

```
func setRows (a: ref[Tensor], b: ref[Tensor], c: ref[Tensor]): ref[Tensor]
```

#### diag

```
func diag (a: ref[Tensor]): ref[Tensor]
```

#### diagMaskInf

```
func diagMaskInf (a: ref[Tensor], nPast: Int): ref[Tensor]
```

#### diagMaskInfInplace

```
func diagMaskInfInplace (a: ref[Tensor], nPast: Int): ref[Tensor]
```

#### diagMaskZero

```
func diagMaskZero (a: ref[Tensor], nPast: Int): ref[Tensor]
```

#### diagMaskZeroInplace

```
func diagMaskZeroInplace (a: ref[Tensor], nPast: Int): ref[Tensor]
```

#### softMax

```
func softMax (a: ref[Tensor]): ref[Tensor]
```

#### rope

```
func rope (a: ref[Tensor], b: ref[Tensor], nDims: Int, mode: Int): ref[Tensor]
```

#### pad

```
func pad (t: ref[Tensor], p0-3: Int): ref[Tensor]
```

#### roll

```
func roll (t: ref[Tensor], shift0-3: Int): ref[Tensor]
```

#### graphCompute

```
func graphCompute (graph: ref[CGraph], nThreads: Int): Status
```

### Tensor

#### name
```
handler this.name: CharsPtr;
```

#### data
```
handler this.data: ptr;
```

#### dataF32
```
handler this.dataF32: ref[array[Float]];
```

#### nElements
```
handler this.nElements: Int[64];
```

#### nRows
```
handler this.nRows: Int[64];
```

#### nBytes
```
handler this.nBytes: ArchWord;
```

#### nBytesPad
```
handler this.nBytesPad: ArchWord;
```

#### setInput

```
func setInput ()
```

#### setOutput

```
func setOutput ()
```

#### setParam

```
func setParam ()
```

#### setLoss

```
func setLoss ()
```

#### backendSet

```
func backendSet (data: ptr, offset: ArchWord, size: ArchWord)
```

#### backendGet

```
func backendGet (data: ptr, offset: ArchWord, size: ArchWord)
```

### Backend

#### isCpu

```
handler this.isCpu: Bool;
```

#### isVk

```
handler this.isVk: Bool;
```

#### load

```
func load (path: CharsPtr): ref[Reg]
```

#### cpuLoad

```
func cpuLoad ();
func cpuLoad (exePath: CharsPtr);
```

Load the backend specific for running inference on the CPU. This should be called before
initialization.
The first form of this function looks for the backend library in the current path as well
as in the Ggml library's path. This would cover both running in JIT mode as well as in
pre-compilation mode, which is enough for most cases, but the second form is available for
exceptional cases. The second form will look only within the provided path.

#### vkLoad

```
func vkLoad ();
func vkLoad (exePath: CharsPtr);
```

Load the backend specific fur running inference through Vulkan API. Use this instead of
`cpuLoad` if you want to run inference on the GPU.
The first form of this function looks for the backend library in the current path as well
as in the Ggml library's path. This would cover both running in JIT mode as well as in
pre-compilation mode, which is enough for most cases, but the second form is available for
exceptional cases. The second form will look only within the provided path.

#### cpuInit

```
func cpuInit (): ref[Backend]
```

#### vkInit

```
func vkInit (devNum: ArchWord): ref[Backend]
```

#### free

```
func free (backend: ref[Backend])
```

#### getDefaultBufferType

```
func getDefaultBufferType (): ref[BackendBufferType]
```

#### cpuSetNThreads

```
func cpuSetNThreads (threads: Int)
```

#### graphCompute

```
func graphCompute (graph: ref[CGraph]): Status
```

#### graphComputeAsync

```
func graphComputeAsync (graph: ref[CGraph]): Status
```

### CGraph

#### plan

```
func plan (nThreads: Int, threadPool: ref[ThreadPool]): CPlan
```

#### compute

```
func compute (cplan: ref[CPlan]): Status
```

#### reset

```
func reset ()
```

#### print

```
func print ()
```

### CPlan

#### workSize
```
def workSize: ArchWord;
```

#### workData
```
def workData: ref[array[Word[8]]];
```

#### nThreads
```
def nThreads: Int;
```

#### threadPool
```
def threadPool: ref[ThreadPool];
```

#### abortCallback
```
def abortCallback: ptr[function(ptr)];
```

#### abortCallbackData
```
def abortCallbackData: ptr;
```

### BackendBuffer

#### free

```
func free (buffer: ref[BackendBuffer])
```

### Gallocr

#### new

```
func new (backendBufferType: ref[BackendBufferType]): ref[Gallocr]
```

#### free

```
func free (gallocr: ref[Gallocr])
```

#### allocGraph

```
func allocGraph (graph: ref[CGraph]): Bool
```

### ThreadPool

* `nThreads`: Int (property)

#### new

```
func new (params: ref[Params]): ref[ThreadPool]
```

#### free

```
func free (pool: ref[ThreadPool])
```

#### pause

```
func pause ()
```

#### resume

```
func resume ()
```

### ThreadPool.SchedPriority Enum

* `LOW` (-1)
* `NORMAL` (0)
* `MEDIUM` (1)
* `HIGH` (2)
* `REALTIME` (3)

### ThreadPool.Params

#### cpuMask
```
def cpuMask: array[Bool, 512];
```

#### nThreads
```
def nThreads: Int;
```

#### prio
```
def prio: Int;
```

#### poll
```
def poll: Word;
```

#### strictCpu
```
def strictCpu: Bool;
```

#### paused
```
def paused: Bool;
```

#### init

```
func init (nThreads: Int)
```

#### getDefault

```
func getDefault (): Params
```

#### match

```
func match (a: ref[Params], b: ref[Params]): Bool
```

## Documentation

For detailed documentation of the underlying functions, refer to the [GGML documentation](https://github.com/ggml-org/ggml).

## License

This binding follows the GGML license (MIT). See the `license` file for details.
