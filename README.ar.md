# جـجمل - روابط مكتبة GGML للغة الأسس
[[English]](README.md)

روابط لغة الأسس لمكتبة [GGML](https://github.com/ggml-org/ggml) للموترات.

## الاستخدام

<div dir=rtl>

```
اشمل "مـحا"؛
مـحا.اشمل_حزمة("Alusus/Ggml@0.2"، "جـجمل.أسس")؛
استخدم جـجمل؛
```

</div>

```
import "Apm";
Apm.importPackage("Alusus/Ggml@0.2");
use Ggml;
```

## الأمثلة

راجع مجلد `Examples/`:
- `مثال.أسس` - ضرب مصفوفات باستخدام السياق
- `مثال_المشغل.أسس` - ضرب مصفوفات باستخدام المشغل (فلكان/المعالج)

## دعم فلكان

لتفعيل مشغل فلكان، حدد متغير البيئة قبل التشغيل:

```bash
export GGML_USE_VULKAN=1
```

## مرجع الواجهة البرمجية

### الدوال العامة

#### هيئ (init)

<div dir=rtl>

```
دالة هيئ(معطيات: مـعطيات_تهيئة): سند[سـياق]؛
```

</div>

```
function init(params: InitParams): ref[Context];
```

#### حرر (free)

<div dir=rtl>

```
دالة حرر(سياق: سند[سـياق])؛
```

</div>

```
function free(ctx: ref[Context]);
```

#### هيئ_نوما (numaInit)

<div dir=rtl>

```
دالة هيئ_نوما(إستراتيجية_نوما: إسـتراتيجية_نوما)؛
```

</div>

```
function numaInit(numaStrategy: NumaStrategy);
```

#### أهو_نوما (isNuma)

<div dir=rtl>

```
دالة أهو_نوما(): ثنائي؛
```

</div>

```
function isNuma(): Bool;
```

#### هات_تكلفة_الموتر (getTensorOverhead)

<div dir=rtl>

```
دالة هات_تكلفة_الموتر(): طـبيعي_متكيف؛
```

</div>

```
function getTensorOverhead(): ArchWord;
```

#### هات_حجم_النوع (getTypeSize)

<div dir=rtl>

```
دالة هات_حجم_النوع(نوع: نـوع): طـبيعي_متكيف؛
```

</div>

```
function getTypeSize(type: Type): ArchWord;
```

#### هات_حجم_الصف (getRowSize)

<div dir=rtl>

```
دالة هات_حجم_الصف(نوع: نـوع، ب_ع: صحيح[64]): طـبيعي_متكيف؛
```

</div>

```
function getRowSize(type: Type, ne: Int[64]): ArchWord;
```

#### هات_تكلفة_البيان (getGraphOverhead)

<div dir=rtl>

```
دالة هات_تكلفة_البيان(): طـبيعي_متكيف؛
```

</div>

```
function getGraphOverhead(): ArchWord;
```

#### حدد_دالة_الإجهاض (setAbortCallback)

<div dir=rtl>

```
دالة حدد_دالة_الإجهاض(د: مؤشر[دالة(مؤشر_محارف)]): مؤشر[دالة(مؤشر_محارف)]؛
```

</div>

```
function setAbortCallback(cb: ptr[func(CharsPtr)]): ptr[func(CharsPtr)];
```

#### أجهض (abort)

<div dir=rtl>

```
دالة أجهض(ملف: مؤشر_محارف، سطر: صحيح، صيغة: مؤشر_محارف، ...أيما)؛
```

</div>

```
function abort(file: CharsPtr, line: Int, fmt: CharsPtr, ...any);
```

#### حول_نوع_بنية_إلى_نوع (fTypeToType)

<div dir=rtl>

```
دالة حول_نوع_بنية_إلى_نوع(نوع_بنية: نـوع_بنية): نـوع؛
```

</div>

```
function fTypeToType(ftype: FType): Type;
```

#### حول_الحالة_لنص (statusToString)

<div dir=rtl>

```
دالة حول_الحالة_لنص(حالة: حـالة): مؤشر_محارف؛
```

</div>

```
function statusToString(status: Status): CharsPtr;
```

#### حول_ع16_إلى_ع32 (fp16ToFp32)

<div dir=rtl>

```
دالة حول_ع16_إلى_ع32(ق: طـبيعي[16]): عائم؛
دالة حول_ع16_إلى_ع32(من: سند[مصفوفة[طـبيعي[16]]]، إلى: سند[مصفوفة[عـائم]]، الطول: صـحيح[64])؛
```

</div>

```
function fp16ToFp32(v: Word[16]): Float;
function fp16ToFp32(src: ref[array[Word[16]]], dst: ref[array[Float]], len: Int[64]);
```

#### حول_ع32_إلى_ع16 (fp32ToFp16)

<div dir=rtl>

```
دالة حول_ع32_إلى_ع16(ق: عائم): طـبيعي[16]؛
دالة حول_ع32_إلى_ع16(من: سند[مصفوفة[عـائم]]، إلى: سند[مصفوفة[طـبيعي[16]]]، الطول: صـحيح[64])؛
```

</div>

```
function fp32ToFp16(v: Float): Word[16];
function fp32ToFp16(src: ref[array[Float]], dst: ref[array[Word[16]]], len: Int[64]);
```

#### حول_ع32_إلى_عد16 (fp32ToBf16)

<div dir=rtl>

```
دالة حول_ع32_إلى_عد16(ق: عائم): طـبيعي[16]؛
دالة حول_ع32_إلى_عد16(من: سند[مصفوفة[عـائم]]، إلى: سند[مصفوفة[طـبيعي[16]]]، الطول: صـحيح[64])؛
```

</div>

```
function fp32ToBf16(v: Float): Word[16];
function fp32ToBf16(src: ref[array[Float]], dst: ref[array[Word[16]]], len: Int[64]);
```

#### حول_عد16_إلى_ع32 (bf16ToFp32)

<div dir=rtl>

```
دالة حول_عد16_إلى_ع32(ق: طـبيعي[16]): عائم؛
دالة حول_عد16_إلى_ع32(من: سند[مصفوفة[طـبيعي[16]]]، إلى: سند[مصفوفة[عـائم]]، الطول: صـحيح[64])؛
```

</div>

```
function bf16ToFp32(v: Word[16]): Float;
function bf16ToFp32(src: ref[array[Word[16]]], dst: ref[array[Float]], len: Int[64]);
```

#### ابن_توسيعا_أماميا (buildForwardExpand)

<div dir=rtl>

```
دالة ابن_توسيعا_أماميا(بيان: سند[بـيان]، موتر: سند[مـوتر])؛
```

</div>

```
function buildForwardExpand(graph: ref[CGraph], tensor: ref[Tensor]);
```

#### ابن_توسيعا_خلفيا (buildBackwardExpand)

<div dir=rtl>

```
دالة ابن_توسيعا_خلفيا(سياق: سند[سـياق]، بيان: سند[بـيان]، موتر: سند[سند[مـوتر]])؛
```

</div>

```
function buildBackwardExpand(context: ref[Context], graph: ref[CGraph], tensor: ref[ref[Tensor]]);
```

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

<div dir=rtl>

```
عرف حجم_الذاكرة: طـبيعي_متكيف؛
```

</div>

```
def memSize: ArchWord;
```

#### صوان_الذاكرة (memBuffer)

<div dir=rtl>

```
عرف صوان_الذاكرة: مؤشر؛
```

</div>

```
def memBuffer: ptr;
```

#### بلا_حجز (noAlloc)

<div dir=rtl>

```
عرف بلا_حجز: ثنائي؛
```

</div>

```
def noAlloc: Bool;
```

### سـياق (Contex)

#### أنشئ_موترا (newTensor)

<div dir=rtl>

```
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0: صحيح[64]): سند[مـوتر]؛
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0: صحيح[64]، ب_ع1: صحيح[64]): سند[مـوتر]؛
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0-2: صحيح[64]): سند[مـوتر]؛
عملية هذا.أنشئ_موترا(نوع: نـوع، ب_ع0-3: صحيح[64]): سند[مـوتر]؛
```

</div>

```
handler this.newTensor(type: Type, ne0: Int[64]): ref[Tensor];
handler this.newTensor(type: Type, ne0: Int[64], ne1: Int[64]): ref[Tensor];
handler this.newTensor(type: Type, ne0-2: Int[64]): ref[Tensor];
handler this.newTensor(type: Type, ne0-3: Int[64]): ref[Tensor];
```

#### كرر (dup)

<div dir=rtl>

```
عملية هذا.كرر(أ: سند[مـوتر]): سند[مـوتر]؛
عملية هذا.كرر(أ: سند[بـيان]، فرض_تدرجات: ثنائي): سند[بـيان]؛
```

</div>

```
handler this.dup(a: ref[Tensor]): ref[Tensor];
handler this.dup(a: ref[CGraph], forceGrads: Bool): ref[CGraph];
```

#### أنشئ_بيانا (newGraph)

<div dir=rtl>

```
عملية هذا.أنشئ_بيانا(): سند[بـيان]؛
عملية هذا.أنشئ_بيانا(حجم: طـبيعي_متكيف، تدرجات: ثنائي): سند[بـيان]؛
```

</div>

```
handler this.newGraph(): ref[CGraph];
handler this.newGraph(size: ArchWord, grads: Bool): ref[CGraph];
```

#### احجز_موترات_مشغل (backendAllocTensors)

<div dir=rtl>

```
عملية هذا.احجز_موترات_مشغل(مشغل: سند[مـشغل]): سند[صـوان_مشغل]؛
```

</div>

```
handler this.backendAllocTensors(backend: ref[Backend]): ref[BackendBuffer];
```

#### اجمع (add)

<div dir=rtl>

```
عملية هذا.اجمع(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.add(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

#### اطرح (sub)

<div dir=rtl>

```
عملية هذا.اطرح(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.sub(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

#### اضرب (mul)

<div dir=rtl>

```
عملية هذا.اضرب(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.mul(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

#### قسم (div)

<div dir=rtl>

```
عملية هذا.قسم(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.div(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

#### لوغاريتم (log)

<div dir=rtl>

```
عملية هذا.لوغاريتم(م: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.log(t: ref[Tensor]): ref[Tensor];
```

#### مجموع (sum)

<div dir=rtl>

```
عملية هذا.مجموع(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.sum(a: ref[Tensor]): ref[Tensor];
```

#### متوسط (mean)

<div dir=rtl>

```
عملية هذا.متوسط(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.mean(a: ref[Tensor]): ref[Tensor];
```

#### أقصى_معامل (argmax)

<div dir=rtl>

```
عملية هذا.أقصى_معامل(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.argmax(a: ref[Tensor]): ref[Tensor];
```

#### معيار (norm)

<div dir=rtl>

```
عملية هذا.معيار(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.norm(a: ref[Tensor]): ref[Tensor];
```

#### معيار_المتوسط (rmsNorm)

<div dir=rtl>

```
عملية هذا.معيار_المتوسط(أ: سند[مـوتر]، إبسلون: عائم): سند[مـوتر]؛
```

</div>

```
handler this.rmsNorm(a: ref[Tensor], eps: Float): ref[Tensor];
```

#### اضرب_مصفوفات (mulMat)

<div dir=rtl>

```
عملية هذا.اضرب_مصفوفات(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.mulMat(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

#### حجم (scale)

<div dir=rtl>

```
عملية هذا.حجم(أ: سند[مـوتر]، ألفا: عائم): سند[مـوتر]؛
```

</div>

```
handler this.scale(a: ref[Tensor], alpha: Float): ref[Tensor];
```

#### حدد (set)

<div dir=rtl>

```
عملية هذا.حدد(أ: سند[مـوتر]، قيمة: عائم): سند[مـوتر]؛
```

</div>

```
handler this.set(a: ref[Tensor], value: Float): ref[Tensor];
```

#### انسخ_إلى (cpy)

<div dir=rtl>

```
عملية هذا.انسخ_إلى(مصدر: سند[مـوتر]، وجهة: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.cpy(src: ref[Tensor], dst: ref[Tensor]): ref[Tensor];
```

#### استمر (cont)

<div dir=rtl>

```
عملية هذا.استمر(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.cont(a: ref[Tensor]): ref[Tensor];
```

#### أعد_التشكيل (reshape)

<div dir=rtl>

```
عملية هذا.أعد_التشكيل(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
عملية هذا.أعد_التشكيل(أ: سند[مـوتر]، ب_ع0: صحيح[64]): سند[مـوتر]؛
عملية هذا.أعد_التشكيل(أ: سند[مـوتر]، ب_ع0-3: صحيح[64]): سند[مـوتر]؛
```

</div>

```
handler this.reshape(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
handler this.reshape(a: ref[Tensor], ne0: Int[64]): ref[Tensor];
handler this.reshape(a: ref[Tensor], ne0-3: Int[64]): ref[Tensor];
```

#### اعرض (view)

<div dir=rtl>

```
عملية هذا.اعرض(أ: سند[مـوتر]، ب_ع0: صحيح[64]، إزاحة: طـبيعي_متكيف): سند[مـوتر]؛
```

</div>

```
handler this.view(a: ref[Tensor], ne0: Int[64], offset: ArchWord): ref[Tensor];
```

#### بدل (permute)

<div dir=rtl>

```
عملية هذا.بدل(أ: سند[مـوتر]، محور1-4: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.permute(a: ref[Tensor], axis1-4: Int): ref[Tensor];
```

#### منقولة (transpose)

<div dir=rtl>

```
عملية هذا.منقولة(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.transpose(a: ref[Tensor]): ref[Tensor];
```

#### هات_صفوفا (getRows)

<div dir=rtl>

```
عملية هذا.هات_صفوفا(أ: سند[مـوتر]، ب: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.getRows(a: ref[Tensor], b: ref[Tensor]): ref[Tensor];
```

#### حدد_صفوفا (setRows)

<div dir=rtl>

```
عملية هذا.حدد_صفوفا(أ: سند[مـوتر]، ب: سند[مـوتر]، ج: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.setRows(a: ref[Tensor], b: ref[Tensor], c: ref[Tensor]): ref[Tensor];
```

#### قطري (diag)

<div dir=rtl>

```
عملية هذا.قطري(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.diag(a: ref[Tensor]): ref[Tensor];
```

#### قناع_قطري_لا_نهائي (diagMaskInf)

<div dir=rtl>

```
عملية هذا.قناع_قطري_لا_نهائي(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.diagMaskInf(a: ref[Tensor], nPast: Int): ref[Tensor];
```

#### قناع_قطري_لا_نهائي_موضعيا (diagMaskInfInplace)

<div dir=rtl>

```
عملية هذا.قناع_قطري_لا_نهائي_موضعيا(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.diagMaskInfInplace(a: ref[Tensor], nPast: Int): ref[Tensor];
```

#### قناع_قطري_صفري (diagMaskZero)

<div dir=rtl>

```
عملية هذا.قناع_قطري_صفري(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.diagMaskZero(a: ref[Tensor], nPast: Int): ref[Tensor];
```

#### قناع_قطري_صفري_موضعيا (diagMaskZeroInplace)

<div dir=rtl>

```
عملية هذا.قناع_قطري_صفري_موضعيا(أ: سند[مـوتر]، ن_سابق: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.diagMaskZeroInplace(a: ref[Tensor], nPast: Int): ref[Tensor];
```

#### أسية_مطبعة (softMax)

<div dir=rtl>

```
عملية هذا.أسية_مطبعة(أ: سند[مـوتر]): سند[مـوتر]؛
```

</div>

```
handler this.softMax(a: ref[Tensor]): ref[Tensor];
```

#### تضمين_موضعي_دوار (rope)

<div dir=rtl>

```
عملية هذا.تضمين_موضعي_دوار(أ: سند[مـوتر]، ب: سند[مـوتر]، ن_أبعاد: صحيح، وضع: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.rope(a: ref[Tensor], b: ref[Tensor], nDims: Int, mode: Int): ref[Tensor];
```

#### حشو (pad)

<div dir=rtl>

```
عملية هذا.حشو(م: سند[مـوتر]، ح0-3: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.pad(t: ref[Tensor], p0-3: Int): ref[Tensor];
```

#### لف (roll)

<div dir=rtl>

```
عملية هذا.لف(م: سند[مـوتر]، إزاحة0-3: صحيح): سند[مـوتر]؛
```

</div>

```
handler this.roll(t: ref[Tensor], shift0-3: Int): ref[Tensor];
```

#### شغل_البيان (graphCompute)

<div dir=rtl>

```
عملية هذا.شغل_البيان(بيان: سند[بـيان]، عدد_المسالك: صحيح): حـالة؛
```

</div>

```
handler this.graphCompute(graph: ref[CGraph], nThreads: Int): Status;
```

### مـوتر (Tensor)

#### الاسم (name)

<div dir=rtl>

```
عملية هذا.الاسم: مؤشر_محارف؛
```

</div>

```
handler this.name: CharsPtr;
```

#### البيانات (data)

<div dir=rtl>

```
عملية هذا.البيانات: مؤشر؛
```

</div>

```
handler this.data: ptr;
```

#### البيانات_ع32 (dataF32)

<div dir=rtl>

```
عملية هذا.البيانات_ع32: سند[مصفوفة[عائم]]؛
```

</div>

```
handler this.dataF32: ref[array[Float]];
```

#### عدد_العناصر (nElements)

<div dir=rtl>

```
عملية هذا.عدد_العناصر: صحيح[64]؛
```

</div>

```
handler this.nElements: Int[64];
```

#### عدد_الصفوف (nRows)

<div dir=rtl>

```
عملية هذا.عدد_الصفوف: صحيح[64]؛
```

</div>

```
handler this.nRows: Int[64];
```

#### عدد_البايتات (nBytes)

<div dir=rtl>

```
عملية هذا.عدد_البايتات: طـبيعي_متكيف؛
```

</div>

```
handler this.nBytes: ArchWord;
```

#### عدد_البايتات_بمحاذاة (nBytesPad)

<div dir=rtl>

```
عملية هذا.عدد_البايتات_بمحاذاة: طـبيعي_متكيف؛
```

</div>

```
handler this.nBytesPad: ArchWord;
```

#### حدد_كمدخل (setInput)

<div dir=rtl>

```
عملية هذا.حدد_كمدخل()؛
```

</div>

```
handler this.setInput();
```

#### حدد_كمخرج (setOutput)

<div dir=rtl>

```
عملية هذا.حدد_كمخرج()؛
```

</div>

```
handler this.setOutput();
```

#### حدد_كمعامل (setParam)

<div dir=rtl>

```
عملية هذا.حدد_كمعامل()؛
```

</div>

```
handler this.setParam();
```

#### حدد_كخسارة (setLoss)

<div dir=rtl>

```
عملية هذا.حدد_كخسارة()؛
```

</div>

```
handler this.setLoss();
```

#### حدد_من_المشغل (backendSet)

<div dir=rtl>

```
عملية هذا.حدد_من_المشغل(بيانات: مؤشر، إزاحة: طـبيعي_متكيف، حجم: طـبيعي_متكيف)؛
```

</div>

```
handler this.backendSet(data: ptr, offset: ArchWord, size: ArchWord);
```

#### هات_من_المشغل (backendGet)

<div dir=rtl>

```
عملية هذا.هات_من_المشغل(بيانات: مؤشر، إزاحة: طـبيعي_متكيف، حجم: طـبيعي_متكيف)؛
```

</div>

```
handler this.backendGet(data: ptr, offset: ArchWord, size: ArchWord);
```

### مـشغل (Backend)

#### حمل (load)

<div dir=rtl>

```
عملية هذا_الصنف.حمل(مسار: مؤشر_محارف): سند[سـجل]؛
```

</div>

```
handler this_type.load(path: CharsPtr): ref[Reg];
```

#### حمل_للمعالج (cpuLoad)

<div dir=rtl>

```
عملية هذا_الصنف.حمل_للمعالج()؛
```

</div>

```
handler this_type.cpuLoad();
```

#### حمل_لفلكان (vkLoad)

<div dir=rtl>

```
عملية هذا_الصنف.حمل_لفلكان()؛
```

</div>

```
handler this_type.vkLoad();
```

#### هيئ_للمعالج (cpuInit)

<div dir=rtl>

```
عملية هذا_الصنف.هيئ_للمعالج(): سند[مـشغل]؛
```

</div>

```
handler this_type.cpuInit(): ref[Backend];
```

#### هيئ_لفلكان (vkInit)

<div dir=rtl>

```
عملية هذا_الصنف.هيئ_لفلكان(رقم_جهاز: طـبيعي_متكيف): سند[مـشغل]؛
```

</div>

```
handler this_type.vkInit(devNum: ArchWord): ref[Backend];
```

#### حرر (free)

<div dir=rtl>

```
عملية هذا_الصنف.حرر(سند[مـشغل])؛
```

</div>

```
handler this_type.free(ref[Backend]);
```

#### هات_نوع_الصوان_المبدئي (getDefaultBufferType)

<div dir=rtl>

```
عملية هذا.هات_نوع_الصوان_المبدئي(): سند[نـوع_صوان_مشغل]؛
```

</div>

```
handler this.getDefaultBufferType(): ref[BackendBufferType];
```

#### أللمعالج (isCpu)

<div dir=rtl>

```
عملية هذا.أللمعالج: ثنائي؛
```

</div>

```
handler this.isCpu: Bool;
```

#### ألفلكان (isVk)

<div dir=rtl>

```
عملية هذا.ألفلكان: ثنائي؛
```

</div>

```
handler this.isVk: Bool;
```

#### حدد_عدد_المسالك_للمعالج (cpuSetNThreads)

<div dir=rtl>

```
عملية هذا.حدد_عدد_المسالك_للمعالج(مسالك: صحيح)؛
```

</div>

```
handler this.cpuSetNThreads(threads: Int);
```

#### شغل_البيان (graphCompute)

<div dir=rtl>

```
عملية هذا.شغل_البيان(بيان: سند[بـيان]): حـالة؛
```

</div>

```
handler this.graphCompute(graph: ref[CGraph]): Status;
```

#### شغل_البيان_بالتوازي (graphComputeAsync)

<div dir=rtl>

```
عملية هذا.شغل_البيان_بالتوازي(بيان: سند[بـيان]): حـالة؛
```

</div>

```
handler this.graphComputeAsync(graph: ref[CGraph]): Status;
```

### بـيان (CGraph)

#### خطط (plan)

<div dir=rtl>

```
عملية هذا.خطط(عدد_المسالك: صحيح، مجمع_مسالك: سند[مـجمع_مسالك]): خـطة؛
```

</div>

```
handler this.plan(nThreads: Int, threadPool: ref[ThreadPool]): CPlan;
```

#### شغل (compute)

<div dir=rtl>

```
عملية هذا.شغل(خطة: سند[خـطة]): حـالة؛
```

</div>

```
handler this.compute(cplan: ref[CPlan]): Status;
```

#### أعد_الضبط (reset)

<div dir=rtl>

```
عملية هذا.أعد_الضبط()؛
```

</div>

```
handler this.reset();
```

#### اطبع (print)

<div dir=rtl>

```
عملية هذا.اطبع()؛
```

</div>

```
handler this.print();
```

### خـطة (CPlan)

#### حجم_العمل (workSize)

<div dir=rtl>

```
عرف حجم_العمل: طـبيعي_متكيف؛
```

</div>

```
def workSize: ArchWord;
```

#### بيانات_العمل (workData)

<div dir=rtl>

```
عرف بيانات_العمل: سند[مصفوفة[طـبيعي[8]]]؛
```

</div>

```
def workData: ref[array[Word[8]]];
```

#### عدد_المسالك (nThreads)

<div dir=rtl>

```
عرف عدد_المسالك: صحيح؛
```

</div>

```
def nThreads: Int;
```

#### مجمع_المسالك (threadPool)

<div dir=rtl>

```
عرف مجمع_المسالك: سند[مـجمع_مسالك]؛
```

</div>

```
def threadPool: ref[ThreadPool];
```

#### دالة_الإجهاض (abortCallback)

<div dir=rtl>

```
عرف دالة_الإجهاض: مؤشر[دالة(مؤشر)]؛
```

</div>

```
def abortCallback: ptr[function(ptr)];
```

#### بيانات_دالة_الإجهاض (abortCallbackData)

<div dir=rtl>

```
عرف بيانات_دالة_الإجهاض: مؤشر؛
```

</div>

```
def abortCallbackData: ptr;
```

### صـوان_مشغل (BackendBuffer)

#### حرر (free)

<div dir=rtl>

```
عملية هذا_الصنف.حرر(سند[صـوان_مشغل])؛
```

</div>

```
handler this_type.free(ref[BackendBuffer]);
```

### حـاجز_بيان (Gallocr)

#### أنشئ (new)

<div dir=rtl>

```
عملية هذا_الصنف.أنشئ(نوع_صوان_مشغل: سند[نـوع_صوان_مشغل]): سند[حـاجز_بيان]؛
```

</div>

```
handler this_type.new(backendBufferType: ref[BackendBufferType]): ref[Gallocr];
```

#### حرر (free)

<div dir=rtl>

```
عملية هذا_الصنف.حرر(سند[حـاجز_بيان])؛
```

</div>

```
handler this_type.free(ref[Gallocr]);
```

#### احجز_بيانا (allocGraph)

<div dir=rtl>

```
عملية هذا.احجز_بيانا(بيان: سند[بـيان]): ثنائي؛
```

</div>

```
handler this.allocGraph(graph: ref[CGraph]): Bool;
```

### مـجمع_مسالك (ThreadPool)

#### أنشئ (new)

<div dir=rtl>

```
عملية هذا_الصنف.أنشئ(معطيات: سند[مـعطيات]): سند[مـجمع_مسالك]؛
```

</div>

```
handler this_type.new(params: ref[Params]): ref[ThreadPool];
```

#### حرر (free)

<div dir=rtl>

```
عملية هذا_الصنف.حرر(مجمع: سند[مـجمع_مسالك])؛
```

</div>

```
handler this_type.free(pool: ref[ThreadPool]);
```

#### عدد_المسالك (nThreads)

<div dir=rtl>

```
عملية هذا.عدد_المسالك: صحيح؛
```

</div>

```
handler this.nThreads: Int;
```

#### أوقف (pause)

<div dir=rtl>

```
عملية هذا.أوقف()؛
```

</div>

```
handler this.pause();
```

#### استأنف (resume)

<div dir=rtl>

```
عملية هذا.استأنف()؛
```

</div>

```
handler this.resume();
```

### سرد أولـوية_جدولة (ThreadPool.SchedPriority Enum)

* `_منخفضة_` (`LOW`) (-1)
* `_عادية_` (`NORMAL`) (0)
* `_متوسطة_` (`MEDIUM`) (1)
* `_عالية_` (`HIGH`) (2)
* `_وقت_حقيقي_` (`REALTIME`) (3)

### مـجمع_مسالك.مـعطيات (ThreadPool.Params)

#### قناع_المعالجات (cpuMask)

<div dir=rtl>

```
عرف قناع_المعالجات: مصفوفة[ثنائي، 512]؛
```

</div>

```
def cpuMask: array[Bool, 512];
```

#### عدد_المسالك (nThreads)

<div dir=rtl>

```
عرف عدد_المسالك: صحيح؛
```

</div>

```
def nThreads: Int;
```

#### الأولوية (prio)

<div dir=rtl>

```
عرف الأولوية: صحيح؛
```

</div>

```
def prio: Int;
```

#### الاستقصاء (poll)

<div dir=rtl>

```
عرف الاستقصاء: طـبيعي؛
```

</div>

```
def poll: Word;
```

#### معالج_صارم (strictCpu)

<div dir=rtl>

```
عرف معالج_صارم: ثنائي؛
```

</div>

```
def strictCpu: Bool;
```

#### متوقف (paused)

<div dir=rtl>

```
عرف متوقف: ثنائي؛
```

</div>

```
def paused: Bool;
```

#### هيئ (init)

<div dir=rtl>

```
عملية هذا.هيئ(عدد_المسالك: صحيح)؛
```

</div>

```
handler this.init(nThreads: Int);
```

#### هات_المبدئية (getDefault)

<div dir=rtl>

```
عملية هذا_الصنف.هات_المبدئية(): مـعطيات؛
```

</div>

```
handler this_type.getDefault(): Params;
```

#### طابق (match)

<div dir=rtl>

```
عملية هذا_الصنف.طابق(أ: سند[مـعطيات]، ب: سند[مـعطيات]): ثنائي؛
```

</div>

```
handler this_type.match(a: ref[Params], b: ref[Params]): Bool;
```

## التوثيق

للتوثيق التفصيلي للدوال الأصلية، راجع [توثيق GGML](https://github.com/ggml-org/ggml).

## الرخصة

هذه الروابط تتبع رخصة GGML (MIT). راجع ملف `license` للتفاصيل.
