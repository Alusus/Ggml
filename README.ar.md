# جـجمل (GGML)
[[English]](README.md)

<div dir=rtl>

روابط لغة الأسس لمكتبة [GGML](https://github.com/ggml-org/ggml) للموترات.

## الاستخدام

```
اشمل "مـحا"؛
مـحا.اشمل_حزمة("Alusus/Ggml@0.2"، "جـجمل.أسس")؛
استخدم جـجمل؛
```

<div dir=ltr>

```
import "Apm";
Apm.importPackage("Alusus/Ggml@0.2");
use Ggml;
```

</div>

## الأمثلة

راجع مجلد `Examples/`:
- `مثال.أسس` - ضرب مصفوفات باستخدام السياق
- `مثال_المشغل.أسس` - ضرب مصفوفات باستخدام المشغل (فلكان/المعالج)

## دعم فلكان

لتفعيل مشغل فلكان، حدد متغير البيئة قبل التشغيل:

<div dir=ltr>

```bash
export GGML_USE_VULKAN=1
```

</div>

## مرجع الواجهة البرمجية

### الدوال العامة

#### هيئ (init)

```
دالة هيئ(معطيات: مـعطيات_تهيئة): سند[سـياق]؛
```

<div dir=ltr>

```
function init(params: InitParams): ref[Context];
```

</div>

#### حرر (free)

```
دالة حرر(سياق: سند[سـياق])؛
```

<div dir=ltr>

```
function free(ctx: ref[Context]);
```

</div>

#### هيئ_نوما (numaInit)

```
دالة هيئ_نوما(إستراتيجية_نوما: إسـتراتيجية_نوما)؛
```

<div dir=ltr>

```
function numaInit(numaStrategy: NumaStrategy);
```

</div>

#### أهو_نوما (isNuma)

```
دالة أهو_نوما(): ثنائي؛
```

<div dir=ltr>

```
function isNuma(): Bool;
```

</div>

#### هات_تكلفة_الموتر (getTensorOverhead)

```
دالة هات_تكلفة_الموتر(): طـبيعي_متكيف؛
```

<div dir=ltr>

```
function getTensorOverhead(): ArchWord;
```

</div>

#### هات_حجم_النوع (getTypeSize)

```
دالة هات_حجم_النوع(نوع: نـوع): طـبيعي_متكيف؛
```

<div dir=ltr>

```
function getTypeSize(type: Type): ArchWord;
```

</div>

#### هات_حجم_الصف (getRowSize)

```
دالة هات_حجم_الصف(نوع: نـوع، ب_ع: صحيح[64]): طـبيعي_متكيف؛
```

<div dir=ltr>

```
function getRowSize(type: Type, ne: Int[64]): ArchWord;
```

</div>

#### هات_تكلفة_البيان (getGraphOverhead)

```
دالة هات_تكلفة_البيان(): طـبيعي_متكيف؛
```

<div dir=ltr>

```
function getGraphOverhead(): ArchWord;
```

</div>

#### حدد_دالة_الإجهاض (setAbortCallback)

```
دالة حدد_دالة_الإجهاض(د: مؤشر[دالة(مؤشر_محارف)]): مؤشر[دالة(مؤشر_محارف)]؛
```

<div dir=ltr>

```
function setAbortCallback(cb: ptr[func(CharsPtr)]): ptr[func(CharsPtr)];
```

</div>

#### أجهض (abort)

```
دالة أجهض(ملف: مؤشر_محارف، سطر: صحيح، صيغة: مؤشر_محارف، ...أيما)؛
```

<div dir=ltr>

```
function abort(file: CharsPtr, line: Int, fmt: CharsPtr, ...any);
```

</div>

#### حول_نوع_بنية_إلى_نوع (fTypeToType)

```
دالة حول_نوع_بنية_إلى_نوع(نوع_بنية: نـوع_بنية): نـوع؛
```

<div dir=ltr>

```
function fTypeToType(ftype: FType): Type;
```

</div>

#### حول_الحالة_لنص (statusToString)

```
دالة حول_الحالة_لنص(حالة: حـالة): مؤشر_محارف؛
```

<div dir=ltr>

```
function statusToString(status: Status): CharsPtr;
```

</div>

#### حول_ع16_إلى_ع32 (fp16ToFp32)

```
دالة حول_ع16_إلى_ع32(ق: طـبيعي[16]): عائم؛
دالة حول_ع16_إلى_ع32(من: سند[مصفوفة[طـبيعي[16]]]، إلى: سند[مصفوفة[عـائم]]، الطول: صـحيح[64])؛
```

<div dir=ltr>

```
function fp16ToFp32(v: Word[16]): Float;
function fp16ToFp32(src: ref[array[Word[16]]], dst: ref[array[Float]], len: Int[64]);
```

</div>

#### حول_ع32_إلى_ع16 (fp32ToFp16)

```
دالة حول_ع32_إلى_ع16(ق: عائم): طـبيعي[16]؛
دالة حول_ع32_إلى_ع16(من: سند[مصفوفة[عـائم]]، إلى: سند[مصفوفة[طـبيعي[16]]]، الطول: صـحيح[64])؛
```

<div dir=ltr>

```
function fp32ToFp16(v: Float): Word[16];
function fp32ToFp16(src: ref[array[Float]], dst: ref[array[Word[16]]], len: Int[64]);
```

</div>

#### حول_ع32_إلى_عد16 (fp32ToBf16)

```
دالة حول_ع32_إلى_عد16(ق: عائم): طـبيعي[16]؛
دالة حول_ع32_إلى_عد16(من: سند[مصفوفة[عـائم]]، إلى: سند[مصفوفة[طـبيعي[16]]]، الطول: صـحيح[64])؛
```

<div dir=ltr>

```
function fp32ToBf16(v: Float): Word[16];
function fp32ToBf16(src: ref[array[Float]], dst: ref[array[Word[16]]], len: Int[64]);
```

</div>

#### حول_عد16_إلى_ع32 (bf16ToFp32)

```
دالة حول_عد16_إلى_ع32(ق: طـبيعي[16]): عائم؛
دالة حول_عد16_إلى_ع32(من: سند[مصفوفة[طـبيعي[16]]]، إلى: سند[مصفوفة[عـائم]]، الطول: صـحيح[64])؛
```

<div dir=ltr>

```
function bf16ToFp32(v: Word[16]): Float;
function bf16ToFp32(src: ref[array[Word[16]]], dst: ref[array[Float]], len: Int[64]);
```

</div>

#### ابن_توسيعا_أماميا (buildForwardExpand)

```
دالة ابن_توسيعا_أماميا(بيان: سند[بـيان]، موتر: سند[مـوتر])؛
```

<div dir=ltr>

```
function buildForwardExpand(graph: ref[CGraph], tensor: ref[Tensor]);
```

</div>

#### ابن_توسيعا_خلفيا (buildBackwardExpand)

```
دالة ابن_توسيعا_خلفيا(سياق: سند[سـياق]، بيان: سند[بـيان]، موتر: سند[سند[مـوتر]])؛
```

<div dir=ltr>

```
function buildBackwardExpand(context: ref[Context], graph: ref[CGraph], tensor: ref[ref[Tensor]]);
```

</div>

### سرد حـالة (Status Enum)

* `_فشل_الحجز_` (`ALLOC_FAILED`) (-2): فشل حجز الذاكرة
* `_فشل_` (`FAILED`) (-1): فشلت العملية
* `_نجاح_` (`SUCCESS`) (0): نجحت العملية
* `_إجهاض_` (`ABORTED`) (1): أُجهضت العملية

### سرد نـوع (Type Enum)

* `_ع32_` (`F32`) (0): عائم 32 بت
* `_ع16_` (`F16`) (1): عائم 16 بت
* `_ك4_0_` (`Q4_0`) (2): مكمم 4 بت (نسخة 0)
* `_ك4_1_` (`Q4_1`) (3): مكمم 4 بت (نسخة 1)
* `_ك5_0_` (`Q5_0`) (6): مكمم 5 بت (نسخة 0)
* `_ك5_1_` (`Q5_1`) (7): مكمم 5 بت (نسخة 1)
* `_ك8_0_` (`Q8_0`) (8): مكمم 8 بت (نسخة 0)
* `_ك8_1_` (`Q8_1`) (9): مكمم 8 بت (نسخة 1)
* `_ك2_ك_` (`Q2_K`) (10): مكمم 2 بت (ك)
* `_ك3_ك_` (`Q3_K`) (11): مكمم 3 بت (ك)
* `_ك4_ك_` (`Q4_K`) (12): مكمم 4 بت (ك)
* `_ك5_ك_` (`Q5_K`) (13): مكمم 5 بت (ك)
* `_ك6_ك_` (`Q6_K`) (14): مكمم 6 بت (ك)
* `_ص8_` (`I8`) (24): صحيح 8 بت
* `_ص16_` (`I16`) (25): صحيح 16 بت
* `_ص32_` (`I32`) (26): صحيح 32 بت
* `_ص64_` (`I64`) (27): صحيح 64 بت
* `_ع64_` (`F64`) (28): عائم 64 بت
* `_عد16_` (`BF16`) (30): عائم دماغي 16 بت

### سرد نـوع_بنية (FType Enum)

* `_مجهول_` (`UNKNOWN`) (-1)
* `_كله_ع32_` (`ALL_F32`) (0)
* `_أغلبه_ع16_` (`MOSTLY_F16`) (1)
* `_أغلبه_ك4_0_` (`MOSTLY_Q4_0`) (2)
* `_أغلبه_ك4_1_` (`MOSTLY_Q4_1`) (3)
* `_أغلبه_ك4_1_وبعض_ع16_` (`MOSTLY_Q4_1_SOME_F16`) (4)
* `_أغلبه_ك8_0_` (`MOSTLY_Q8_0`) (7)
* `_أغلبه_ك5_0_` (`MOSTLY_Q5_0`) (8)
* `_أغلبه_ك5_1_` (`MOSTLY_Q5_1`) (9)
* `_أغلبه_ك2_ك_` (`MOSTLY_Q2_K`) (10)
* `_أغلبه_ك3_ك_` (`MOSTLY_Q3_K`) (11)
* `_أغلبه_ك4_ك_` (`MOSTLY_Q4_K`) (12)
* `_أغلبه_ك5_ك_` (`MOSTLY_Q5_K`) (13)
* `_أغلبه_ك6_ك_` (`MOSTLY_Q6_K`) (14)
* `_أغلبه_عد16_` (`MOSTLY_BF16`) (24)

### سرد إسـتراتيجية_نوما (NumaStrategy Enum)

* `_معطل_` (`DISABLED`) (0)
* `_توزيع_` (`DISTRIBUTE`) (1)
* `_عزل_` (`ISOLATE`) (2)
* `_تحكم_نوما_` (`NUMACTL`) (3)
* `_مرآة_` (`MIRROR`) (4)
* `_عدد_` (`COUNT`) (5)

### مـعطيات_تهيئة (InitParams)

#### حجم_الذاكرة (memSize)

```
عرف حجم_الذاكرة: طـبيعي_متكيف؛
```

<div dir=ltr>

```
def memSize: ArchWord;
```

</div>

#### صوان_الذاكرة (memBuffer)

```
عرف صوان_الذاكرة: مؤشر؛
```

<div dir=ltr>

```
def memBuffer: ptr;
```

</div>

#### بلا_حجز (noAlloc)

```
عرف بلا_حجز: ثنائي؛
```

<div dir=ltr>

```
def noAlloc: Bool;
```

</div>

### سـياق (Contex)

#### أنشئ_موترا (newTensor)

```
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0: صحيح[64]): سند[مـوتر]؛
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0: صحيح[64]، ب_ع1: صحيح[64]): سند[مـوتر]؛
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0-2: صحيح[64]): سند[مـوتر]؛
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0-3: صحيح[64]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.newTensor(type: Type, ne0: Int[64]): ref[Tensor];
handler this.newTensor(type: Type, ne0: Int[64], ne1: Int[64]): ref[Tensor];
handler this.newTensor(type: Type, ne0-2: Int[64]): ref[Tensor];
handler this.newTensor(type: Type, ne0-3: Int[64]): ref[Tensor];
```

</div>

#### كرر (dup)

```
عملية هذا.كرر(أ: سند[مـوتر]): سند[مـوتر]؛
عملية هذا.كرر(أ: سند[بـيان]، فرض_تدرجات: ثنائي): سند[بـيان]؛
```

<div dir=ltr>

```
handler this.dup(a: ref[Tensor]): ref[Tensor];
handler this.dup(a: ref[CGraph], forceGrads: Bool): ref[CGraph];
```

</div>

#### أنشئ_بيانا (newGraph)

```
عملية هذا.أنشئ_بيانا(): سند[بـيان]؛
عملية هذا.أنشئ_بيانا(حجم: طـبيعي_متكيف، تدرجات: ثنائي): سند[بـيان]؛
```

<div dir=ltr>

```
handler this.newGraph(): ref[CGraph];
handler this.newGraph(size: ArchWord, grads: Bool): ref[CGraph];
```

</div>

#### احجز_موترات_مشغل (backendAllocTensors)

```
عملية هذا.احجز_موترات_مشغل(مشغل: سند[مـشغل]): سند[صـوان_مشغل]؛
```

<div dir=ltr>

```
handler this.backendAllocTensors(backend: ref[Backend]): ref[BackendBuffer];
```

</div>

#### اجمع (add)

```
عملية هذا.اجمع(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.add(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

</div>

#### اطرح (sub)

```
عملية هذا.اطرح(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.sub(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

</div>

#### اضرب (mul)

```
عملية هذا.اضرب(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.mul(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

</div>

#### قسم (div)

```
عملية هذا.قسم(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.div(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

</div>

#### لوغاريتم (log)

```
عملية هذا.لوغاريتم(م: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.log(t: ref[Tensor]): ref[Tensor];
```

</div>

#### مجموع (sum)

```
عملية هذا.مجموع(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.sum(a: ref[Tensor]): ref[Tensor];
```

</div>

#### متوسط (mean)

```
عملية هذا.متوسط(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.mean(a: ref[Tensor]): ref[Tensor];
```

</div>

#### أقصى_معامل (argmax)

```
عملية هذا.أقصى_معامل(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.argmax(a: ref[Tensor]): ref[Tensor];
```

</div>

#### معيار (norm)

```
عملية هذا.معيار(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.norm(a: ref[Tensor]): ref[Tensor];
```

</div>

#### معيار_المتوسط (rmsNorm)

```
عملية هذا.معيار_المتوسط(أ: سند[مـوتر]، إبسلون: عائم): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.rmsNorm(a: ref[Tensor], eps: Float): ref[Tensor];
```

</div>

#### اضرب_مصفوفات (mulMat)

```
عملية هذا.اضرب_مصفوفات(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.mulMat(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

</div>

#### حجم (scale)

```
عملية هذا.حجم(أ: سند[مـوتر]، ألفا: عائم): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.scale(a: ref[Tensor], alpha: Float): ref[Tensor];
```

</div>

#### حدد (set)

```
عملية هذا.حدد(أ: سند[مـوتر]، قيمة: عائم): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.set(a: ref[Tensor], value: Float): ref[Tensor];
```

</div>

#### انسخ_إلى (cpy)

```
عملية هذا.انسخ_إلى(مصدر: سند[مـوتر]، وجهة: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.cpy(src: ref[Tensor], dst: ref[Tensor]): ref[Tensor];
```

</div>

#### استمر (cont)

```
عملية هذا.استمر(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.cont(a: ref[Tensor]): ref[Tensor];
```

</div>

#### أعد_التشكيل (reshape)

```
عملية هذا.أعد_التشكيل(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
عملية هذا.أعد_التشكيل(أ: سند[مـوتر]، ب_ع0: صحيح[64]): سند[مـوتر]؛
عملية هذا.أعد_التشكيل(أ: سند[مـوتر]، ب_ع0-3: صحيح[64]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.reshape(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
handler this.reshape(a: ref[Tensor], ne0: Int[64]): ref[Tensor];
handler this.reshape(a: ref[Tensor], ne0-3: Int[64]): ref[Tensor];
```

</div>

#### اعرض (view)

```
عملية هذا.اعرض(أ: سند[مـوتر]، ب_ع0: صحيح[64]، إزاحة: طـبيعي_متكيف): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.view(a: ref[Tensor], ne0: Int[64], offset: ArchWord): ref[Tensor];
```

</div>

#### بدل (permute)

```
عملية هذا.بدل(أ: سند[مـوتر]، محور1-4: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.permute(a: ref[Tensor], axis1-4: Int): ref[Tensor];
```

</div>

#### منقولة (transpose)

```
عملية هذا.منقولة(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.transpose(a: ref[Tensor]): ref[Tensor];
```

</div>

#### هات_صفوفا (getRows)

```
عملية هذا.هات_صفوفا(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.getRows(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

</div>

#### حدد_صفوفا (setRows)

```
عملية هذا.حدد_صفوفا(أ: سند[مـوتر]، ب: سند[مـوتر]، ج: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.setRows(a: ref[Tensor], b: ref[Tensor], c: ref[Tensor]): ref[Tensor];
```

</div>

#### قطري (diag)

```
عملية هذا.قطري(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.diag(a: ref[Tensor]): ref[Tensor];
```

</div>

#### قناع_قطري_لا_نهائي (diagMaskInf)

```
عملية هذا.قناع_قطري_لا_نهائي(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.diagMaskInf(a: ref[Tensor], nPast: Int): ref[Tensor];
```

</div>

#### قناع_قطري_لا_نهائي_موضعيا (diagMaskInfInplace)

```
عملية هذا.قناع_قطري_لا_نهائي_موضعيا(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.diagMaskInfInplace(a: ref[Tensor], nPast: Int): ref[Tensor];
```

</div>

#### قناع_قطري_صفري (diagMaskZero)

```
عملية هذا.قناع_قطري_صفري(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.diagMaskZero(a: ref[Tensor], nPast: Int): ref[Tensor];
```

</div>

#### قناع_قطري_صفري_موضعيا (diagMaskZeroInplace)

```
عملية هذا.قناع_قطري_صفري_موضعيا(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.diagMaskZeroInplace(a: ref[Tensor], nPast: Int): ref[Tensor];
```

</div>

#### أسية_مطبعة (softMax)

```
عملية هذا.أسية_مطبعة(أ: سند[مـوتر]): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.softMax(a: ref[Tensor]): ref[Tensor];
```

</div>

#### تضمين_موضعي_دوار (rope)

```
عملية هذا.تضمين_موضعي_دوار(أ: سند[مـوتر]، ب: سند[مـوتر]، ن_أبعاد: صحيح، وضع: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.rope(a: ref[Tensor], b: ref[Tensor], nDims: Int, mode: Int): ref[Tensor];
```

</div>

#### حشو (pad)

```
عملية هذا.حشو(م: سند[مـوتر]، ح0-3: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.pad(t: ref[Tensor], p0-3: Int): ref[Tensor];
```

</div>

#### لف (roll)

```
عملية هذا.لف(م: سند[مـوتر]، إزاحة0-3: صحيح): سند[مـوتر]؛
```

<div dir=ltr>

```
handler this.roll(t: ref[Tensor], shift0-3: Int): ref[Tensor];
```

</div>

#### شغل_البيان (graphCompute)

```
عملية هذا.شغل_البيان(بيان: سند[بـيان]، عدد_المسالك: صحيح): حـالة؛
```

<div dir=ltr>

```
handler this.graphCompute(graph: ref[CGraph], nThreads: Int): Status;
```

</div>

### مـوتر (Tensor)

#### الاسم (name)

```
عملية هذا.الاسم: مؤشر_محارف؛
```

<div dir=ltr>

```
handler this.name: CharsPtr;
```

</div>

#### البيانات (data)

```
عملية هذا.البيانات: مؤشر؛
```

<div dir=ltr>

```
handler this.data: ptr;
```

</div>

#### البيانات_ع32 (dataF32)

```
عملية هذا.البيانات_ع32: سند[مصفوفة[عائم]]؛
```

<div dir=ltr>

```
handler this.dataF32: ref[array[Float]];
```

</div>

#### عدد_العناصر (nElements)

```
عملية هذا.عدد_العناصر: صحيح[64]؛
```

<div dir=ltr>

```
handler this.nElements: Int[64];
```

</div>

#### عدد_الصفوف (nRows)

```
عملية هذا.عدد_الصفوف: صحيح[64]؛
```

<div dir=ltr>

```
handler this.nRows: Int[64];
```

</div>

#### عدد_البايتات (nBytes)

```
عملية هذا.عدد_البايتات: طـبيعي_متكيف؛
```

<div dir=ltr>

```
handler this.nBytes: ArchWord;
```

</div>

#### عدد_البايتات_بمحاذاة (nBytesPad)

```
عملية هذا.عدد_البايتات_بمحاذاة: طـبيعي_متكيف؛
```

<div dir=ltr>

```
handler this.nBytesPad: ArchWord;
```

</div>

#### حدد_كمدخل (setInput)

```
عملية هذا.حدد_كمدخل()؛
```

<div dir=ltr>

```
handler this.setInput();
```

</div>

#### حدد_كمخرج (setOutput)

```
عملية هذا.حدد_كمخرج()؛
```

<div dir=ltr>

```
handler this.setOutput();
```

</div>

#### حدد_كمعامل (setParam)

```
عملية هذا.حدد_كمعامل()؛
```

<div dir=ltr>

```
handler this.setParam();
```

</div>

#### حدد_كخسارة (setLoss)

```
عملية هذا.حدد_كخسارة()؛
```

<div dir=ltr>

```
handler this.setLoss();
```

</div>

#### حدد_من_المشغل (backendSet)

```
عملية هذا.حدد_من_المشغل(بيانات: مؤشر، إزاحة: طـبيعي_متكيف، حجم: طـبيعي_متكيف)؛
```

<div dir=ltr>

```
handler this.backendSet(data: ptr, offset: ArchWord, size: ArchWord);
```

</div>

#### هات_من_المشغل (backendGet)

```
عملية هذا.هات_من_المشغل(بيانات: مؤشر، إزاحة: طـبيعي_متكيف، حجم: طـبيعي_متكيف)؛
```

<div dir=ltr>

```
handler this.backendGet(data: ptr, offset: ArchWord, size: ArchWord);
```

</div>

### مـشغل (Backend)

#### حمل (load)

```
عملية هذا_الصنف.حمل(مسار: مؤشر_محارف): سند[سـجل]؛
```

<div dir=ltr>

```
handler this_type.load(path: CharsPtr): ref[Reg];
```

</div>

#### حمل_للمعالج (cpuLoad)

```
عملية هذا_الصنف.حمل_للمعالج()؛
```

<div dir=ltr>

```
handler this_type.cpuLoad();
```

</div>

#### حمل_لفلكان (vkLoad)

```
عملية هذا_الصنف.حمل_لفلكان()؛
```

<div dir=ltr>

```
handler this_type.vkLoad();
```

</div>

#### هيئ_للمعالج (cpuInit)

```
عملية هذا_الصنف.هيئ_للمعالج(): سند[مـشغل]؛
```

<div dir=ltr>

```
handler this_type.cpuInit(): ref[Backend];
```

</div>

#### هيئ_لفلكان (vkInit)

```
عملية هذا_الصنف.هيئ_لفلكان(رقم_جهاز: طـبيعي_متكيف): سند[مـشغل]؛
```

<div dir=ltr>

```
handler this_type.vkInit(devNum: ArchWord): ref[Backend];
```

</div>

#### حرر (free)

```
عملية هذا_الصنف.حرر(سند[مـشغل])؛
```

<div dir=ltr>

```
handler this_type.free(ref[Backend]);
```

</div>

#### هات_نوع_الصوان_المبدئي (getDefaultBufferType)

```
عملية هذا.هات_نوع_الصوان_المبدئي(): سند[نـوع_صوان_مشغل]؛
```

<div dir=ltr>

```
handler this.getDefaultBufferType(): ref[BackendBufferType];
```

</div>

#### أللمعالج (isCpu)

```
عملية هذا.أللمعالج: ثنائي؛
```

<div dir=ltr>

```
handler this.isCpu: Bool;
```

</div>

#### ألفلكان (isVk)

```
عملية هذا.ألفلكان: ثنائي؛
```

<div dir=ltr>

```
handler this.isVk: Bool;
```

</div>

#### حدد_عدد_المسالك_للمعالج (cpuSetNThreads)

```
عملية هذا.حدد_عدد_المسالك_للمعالج(مسالك: صحيح)؛
```

<div dir=ltr>

```
handler this.cpuSetNThreads(threads: Int);
```

</div>

#### شغل_البيان (graphCompute)

```
عملية هذا.شغل_البيان(بيان: سند[بـيان]): حـالة؛
```

<div dir=ltr>

```
handler this.graphCompute(graph: ref[CGraph]): Status;
```

</div>

#### شغل_البيان_بالتوازي (graphComputeAsync)

```
عملية هذا.شغل_البيان_بالتوازي(بيان: سند[بـيان]): حـالة؛
```

<div dir=ltr>

```
handler this.graphComputeAsync(graph: ref[CGraph]): Status;
```

</div>

### بـيان (CGraph)

#### خطط (plan)

```
عملية هذا.خطط(عدد_المسالك: صحيح، مجمع_مسالك: سند[مـجمع_مسالك]): خـطة؛
```

<div dir=ltr>

```
handler this.plan(nThreads: Int, threadPool: ref[ThreadPool]): CPlan;
```

</div>

#### شغل (compute)

```
عملية هذا.شغل(خطة: سند[خـطة]): حـالة؛
```

<div dir=ltr>

```
handler this.compute(cplan: ref[CPlan]): Status;
```

</div>

#### أعد_الضبط (reset)

```
عملية هذا.أعد_الضبط()؛
```

<div dir=ltr>

```
handler this.reset();
```

</div>

#### اطبع (print)

```
عملية هذا.اطبع()؛
```

<div dir=ltr>

```
handler this.print();
```

</div>

### خـطة (CPlan)

#### حجم_العمل (workSize)

```
عرف حجم_العمل: طـبيعي_متكيف؛
```

<div dir=ltr>

```
def workSize: ArchWord;
```

</div>

#### بيانات_العمل (workData)

```
عرف بيانات_العمل: سند[مصفوفة[طـبيعي[8]]]؛
```

<div dir=ltr>

```
def workData: ref[array[Word[8]]];
```

</div>

#### عدد_المسالك (nThreads)

```
عرف عدد_المسالك: صحيح؛
```

<div dir=ltr>

```
def nThreads: Int;
```

</div>

#### مجمع_المسالك (threadPool)

```
عرف مجمع_المسالك: سند[مـجمع_مسالك]؛
```

<div dir=ltr>

```
def threadPool: ref[ThreadPool];
```

</div>

#### دالة_الإجهاض (abortCallback)

```
عرف دالة_الإجهاض: مؤشر[دالة(مؤشر)]؛
```

<div dir=ltr>

```
def abortCallback: ptr[function(ptr)];
```

</div>

#### بيانات_دالة_الإجهاض (abortCallbackData)

```
عرف بيانات_دالة_الإجهاض: مؤشر؛
```

<div dir=ltr>

```
def abortCallbackData: ptr;
```

</div>

### صـوان_مشغل (BackendBuffer)

#### حرر (free)

```
عملية هذا_الصنف.حرر(سند[صـوان_مشغل])؛
```

<div dir=ltr>

```
handler this_type.free(ref[BackendBuffer]);
```

</div>

### حـاجز_بيان (Gallocr)

#### أنشئ (new)

```
عملية هذا_الصنف.أنشئ(نوع_صوان_مشغل: سند[نـوع_صوان_مشغل]): سند[حـاجز_بيان]؛
```

<div dir=ltr>

```
handler this_type.new(backendBufferType: ref[BackendBufferType]): ref[Gallocr];
```

</div>

#### حرر (free)

```
عملية هذا_الصنف.حرر(سند[حـاجز_بيان])؛
```

<div dir=ltr>

```
handler this_type.free(ref[Gallocr]);
```

</div>

#### احجز_بيانا (allocGraph)

```
عملية هذا.احجز_بيانا(بيان: سند[بـيان]): ثنائي؛
```

<div dir=ltr>

```
handler this.allocGraph(graph: ref[CGraph]): Bool;
```

</div>

### مـجمع_مسالك (ThreadPool)

#### أنشئ (new)

```
عملية هذا_الصنف.أنشئ(معطيات: سند[مـعطيات]): سند[مـجمع_مسالك]؛
```

<div dir=ltr>

```
handler this_type.new(params: ref[Params]): ref[ThreadPool];
```

</div>

#### حرر (free)

```
عملية هذا_الصنف.حرر(مجمع: سند[مـجمع_مسالك])؛
```

<div dir=ltr>

```
handler this_type.free(pool: ref[ThreadPool]);
```

</div>

#### عدد_المسالك (nThreads)

```
عملية هذا.عدد_المسالك: صحيح؛
```

<div dir=ltr>

```
handler this.nThreads: Int;
```

</div>

#### أوقف (pause)

```
عملية هذا.أوقف()؛
```

<div dir=ltr>

```
handler this.pause();
```

</div>

#### استأنف (resume)

```
عملية هذا.استأنف()؛
```

<div dir=ltr>

```
handler this.resume();
```

</div>

### سرد أولـوية_جدولة (ThreadPool.SchedPriority Enum)

* `_منخفضة_` (`LOW`) (-1)
* `_عادية_` (`NORMAL`) (0)
* `_متوسطة_` (`MEDIUM`) (1)
* `_عالية_` (`HIGH`) (2)
* `_وقت_حقيقي_` (`REALTIME`) (3)

### مـجمع_مسالك.مـعطيات (ThreadPool.Params)

#### قناع_المعالجات (cpuMask)

```
عرف قناع_المعالجات: مصفوفة[ثنائي، 512]؛
```

<div dir=ltr>

```
def cpuMask: array[Bool, 512];
```

</div>

#### عدد_المسالك (nThreads)

```
عرف عدد_المسالك: صحيح؛
```

<div dir=ltr>

```
def nThreads: Int;
```

</div>

#### الأولوية (prio)

```
عرف الأولوية: صحيح؛
```

<div dir=ltr>

```
def prio: Int;
```

</div>

#### الاستقصاء (poll)

```
عرف الاستقصاء: طـبيعي؛
```

<div dir=ltr>

```
def poll: Word;
```

</div>

#### معالج_صارم (strictCpu)

```
عرف معالج_صارم: ثنائي؛
```

<div dir=ltr>

```
def strictCpu: Bool;
```

</div>

#### متوقف (paused)

```
عرف متوقف: ثنائي؛
```

<div dir=ltr>

```
def paused: Bool;
```

</div>

#### هيئ (init)

```
عملية هذا.هيئ(عدد_المسالك: صحيح)؛
```

<div dir=ltr>

```
handler this.init(nThreads: Int);
```

</div>

#### هات_المبدئية (getDefault)

```
عملية هذا_الصنف.هات_المبدئية(): مـعطيات؛
```

<div dir=ltr>

```
handler this_type.getDefault(): Params;
```

</div>

#### طابق (match)

```
عملية هذا_الصنف.طابق(أ: سند[مـعطيات]، ب: سند[مـعطيات]): ثنائي؛
```

<div dir=ltr>

```
handler this_type.match(a: ref[Params], b: ref[Params]): Bool;
```

</div>

## التوثيق

للتوثيق التفصيلي للدوال الأصلية، راجع [توثيق GGML](https://github.com/ggml-org/ggml).

## الرخصة

هذه الروابط تتبع رخصة GGML (MIT). راجع ملف `license` للتفاصيل.

</div>

