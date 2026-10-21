"use strict";

const translations = {
  en: {
    skip: "Skip to content", navOverview: "Overview", navFramework: "Framework", navTasks: "Benchmarks", navResources: "Resources", getModel: "Get the model",
    overviewTitle: "Overview",
    overviewDescription: "PhaGen is a 150M-parameter model for generating and analyzing bacteriophage DNA. It learns by reconstructing masked DNA and generates sequences by filling masked positions over multiple steps.",
    overviewDetails: "The same pretrained model provides sequence embeddings and masked-region scores for downstream prediction. The repository includes model code, usage examples, and training code based on VeOmni.",
    metricParameters: "Parameters", metricContext: "Token context", metricEmbedding: "Embedding dimensions", metricBenchmarks: "Prediction benchmarks",
    frameworkTitle: "Model framework",
    frameworkDescription: "PhaGen uses three hierarchical stages to combine nucleotide-level information with broader sequence context. Figure 1 summarizes the training data, masked-diffusion architecture, and evaluated tasks.",
    generationTitle: "How DNA is generated", generationPlay: "Play", generationPause: "Pause", generationReplay: "Replay", generationReset: "Reset",
    generationIntro: "Both examples start from the same 4-nucleotide prefix. M denotes a masked position; blank cells are not yet generated.",
    generationDiffusion: "Masked diffusion", generationAutoregressive: "Autoregressive",
    generationPhagenText: "Fills multiple masked positions per iteration using bidirectional context.",
    generationEvoText: "Reads the existing prefix and appends one nucleotide at a time, from left to right.",
    generationIterations: "Illustrated iterations", generationNewBases: "New bases", generationReady: "Ready", generationPlaying: "Playing", generationPaused: "Paused", generationComplete: "Complete",
    generationNote: "Illustrative sequence and iterations; animation timing does not represent inference time.",
    figureCaption: "Figure 1. Training data, model architecture, and generation and prediction tasks.", figureHint: "Click the figure to enlarge.", figureTitle: "Figure 1 · Overview of PhaGen",
    benchmarksTitle: "Prediction benchmarks", benchmarksDescription: "The pretrained model is evaluated on seven sequence-understanding tasks using sequence scores or embeddings.", taskColumn: "Task", evaluationColumn: "Evaluation",
    task1: "Gene essentiality", task1Text: "Zero-shot prediction for lambda and P1 genes.",
    task2: "T7 SNP mutability", task2Text: "Predict single-nucleotide mutation patterns across 59 T7 genes.",
    task3: "Phage promoter recognition", task3Text: "Classify promoter and non-promoter DNA windows.",
    task4: "Virion-protein subtypes", task4Text: "Predict seven subtypes from protein-coding DNA.",
    task5: "Host prediction", task5Text: "Predict host species from tail-protein coding DNA.",
    task6: "infA mutation fitness", task6Text: "Predict fitness from E. coli infA variant sequences.",
    task7: "TEM-1 β-lactamase fitness", task7Text: "Predict fitness from TEM-1 variant coding sequences.",
    quickstartTitle: "Quick start", quickstartDescription: "Install PhaGen from the repository root. The examples support a downloaded checkpoint or a local model directory.",
    requirements: "Python 3.10+ · PyTorch 2.5 · Transformers 4.54.1", fullDocs: "See the GitHub README for generation and embedding examples.",
    copy: "Copy", copied: "Code copied", copyFailed: "Select and copy the code manually.",
    resourcesTitle: "Resources", resourceCodeText: "Model implementation, examples, VeOmni training, and evaluation code.", resourceModelText: "Pretrained weights, model configuration, and tokenizer files.", resourceDataText: "Evaluation data, formal results, plotting code, and figures.",
    authorsTitle: "Authors", authorsAffiliation: "Data Science and Analytics Thrust, Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511455, China.", correspondingAuthor: "* Corresponding author:",
    footerContact: "Project contact:", footerBuilt: "Training framework:", footerLicense: "License and attribution", footerArt: "Cover illustration generated with AI."
  },
  zh: {
    skip: "跳转到正文", navOverview: "项目简介", navFramework: "模型框架", navTasks: "评测任务", navResources: "开放资源", getModel: "获取模型",
    overviewTitle: "项目简介",
    overviewDescription: "PhaGen 是一个约 1.5 亿参数的噬菌体 DNA 模型，用于生成和分析序列。模型通过重建被遮盖的 DNA 学习，并经过多轮预测生成新序列。",
    overviewDetails: "同一个预训练模型可提供序列表征和遮盖区域评分，用于下游预测。仓库包含模型实现、使用示例，以及基于 VeOmni 的训练代码。",
    metricParameters: "模型参数", metricContext: "上下文词元数", metricEmbedding: "序列表征维度", metricBenchmarks: "预测评测任务",
    frameworkTitle: "模型框架",
    frameworkDescription: "PhaGen 采用三阶段分层架构，结合单个碱基与更大范围的序列信息。论文图 1 展示训练数据、遮盖扩散架构，以及生成和预测评测任务。",
    generationTitle: "DNA 是怎样生成的", generationPlay: "播放", generationPause: "暂停", generationReplay: "重播", generationReset: "重置",
    generationIntro: "两个示例使用相同的 4 个碱基作为前缀。M 表示遮盖位置；空白格表示尚未生成的位置。",
    generationDiffusion: "遮盖扩散", generationAutoregressive: "自回归",
    generationPhagenText: "每轮结合双向上下文，在多个遮盖位置填入碱基。",
    generationEvoText: "读取已有前缀，从左到右逐个追加碱基。",
    generationIterations: "示意迭代", generationNewBases: "新增碱基", generationReady: "待播放", generationPlaying: "播放中", generationPaused: "已暂停", generationComplete: "已完成",
    generationNote: "序列与迭代次数仅作机制示意；播放时间不代表模型推理耗时。",
    figureCaption: "图 1. 训练数据、模型架构，以及生成和预测评测任务。", figureHint: "点击图片可放大。", figureTitle: "图 1 · PhaGen 整体框架",
    benchmarksTitle: "预测评测任务", benchmarksDescription: "通过序列评分或表征，同一个预训练模型用于七项序列理解任务的评测。", taskColumn: "任务", evaluationColumn: "评测内容",
    task1: "基因必需性", task1Text: "对 lambda 和 P1 噬菌体基因进行零样本预测。",
    task2: "T7 单核苷酸突变性", task2Text: "预测 T7 噬菌体 59 个基因中的单核苷酸突变规律。",
    task3: "噬菌体启动子识别", task3Text: "对启动子与非启动子 DNA 窗口进行分类。",
    task4: "病毒粒子蛋白亚型", task4Text: "根据蛋白编码 DNA 预测七类亚型。",
    task5: "宿主预测", task5Text: "根据尾部蛋白编码 DNA 预测宿主物种。",
    task6: "infA 突变适应度", task6Text: "根据大肠杆菌 infA 变体序列预测适应度。",
    task7: "TEM-1 β-内酰胺酶适应度", task7Text: "根据 TEM-1 变体编码序列预测适应度。",
    quickstartTitle: "快速开始", quickstartDescription: "在仓库根目录安装 PhaGen。示例支持加载已下载的模型权重或本地模型目录。",
    requirements: "Python 3.10+ · PyTorch 2.5 · Transformers 4.54.1", fullDocs: "生成 DNA 和提取表征的示例见 GitHub README。",
    copy: "复制", copied: "代码已复制", copyFailed: "请手动选中并复制代码。",
    resourcesTitle: "开放资源", resourceCodeText: "模型实现、使用示例、基于 VeOmni 的训练代码和评测代码。", resourceModelText: "预训练权重、模型配置和分词器文件。", resourceDataText: "评测数据、正式结果、绘图代码和图件。",
    authorsTitle: "作者", authorsAffiliation: "香港科技大学（广州）数据科学与分析学域，中国广州，511455。", correspondingAuthor: "* 通讯作者：",
    footerContact: "项目联系：", footerBuilt: "训练框架：", footerLicense: "许可证与归属声明", footerArt: "封面插画由 AI 生成。"
  }
};

let language = "en";
let toastTimer;
const languageButton = document.getElementById("language-button");
const menuButton = document.getElementById("menu-button");
const mainNav = document.getElementById("main-nav");
const figureDialog = document.getElementById("figure-dialog");

function setLanguage(nextLanguage) {
  language = nextLanguage === "zh" ? "zh" : "en";
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach(element => {
    const value = translations[language][element.dataset.i18n];
    if (value) element.textContent = value;
  });
  languageButton.textContent = language === "en" ? "中文" : "EN";
  languageButton.setAttribute("aria-label", language === "en" ? "切换至中文" : "Switch to English");
  menuButton.setAttribute("aria-label", language === "en" ? "Toggle menu" : "展开或关闭导航");
  document.getElementById("figure-button").setAttribute("aria-label", language === "en" ? "Enlarge Figure 1" : "放大论文图 1");
  document.getElementById("close-figure").setAttribute("aria-label", language === "en" ? "Close figure" : "关闭图件");
  document.title = language === "en" ? "PhaGen — A unified model for bacteriophage DNA" : "PhaGen — 噬菌体 DNA 统一模型";
  updateGenerationControls();
  try { localStorage.setItem("phagen-language", language); } catch (_) {}
}

languageButton.addEventListener("click", () => setLanguage(language === "en" ? "zh" : "en"));

function closeMenu() {
  mainNav.classList.remove("open");
  menuButton.setAttribute("aria-expanded", "false");
}

menuButton.addEventListener("click", () => {
  const open = menuButton.getAttribute("aria-expanded") !== "true";
  mainNav.classList.toggle("open", open);
  menuButton.setAttribute("aria-expanded", String(open));
});
mainNav.querySelectorAll("a").forEach(link => link.addEventListener("click", closeMenu));
document.addEventListener("keydown", event => { if (event.key === "Escape") closeMenu(); });
document.addEventListener("click", event => { if (!event.target.closest(".site-header")) closeMenu(); });
window.addEventListener("resize", () => { if (window.innerWidth > 600) closeMenu(); });

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2400);
}

const copyButton = document.getElementById("copy-button");
copyButton.addEventListener("click", async () => {
  const text = document.getElementById("code-content").textContent;
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(text);
    showToast(translations[language].copied);
  } catch (_) {
    const input = document.createElement("textarea");
    input.value = text;
    input.setAttribute("readonly", "");
    input.style.cssText = "position:fixed;left:-9999px;top:0";
    document.body.append(input);
    input.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } catch (_) {}
    input.remove();
    copyButton.focus({ preventScroll: true });
    showToast(translations[language][copied ? "copied" : "copyFailed"]);
  }
});

document.getElementById("figure-button").addEventListener("click", () => {
  figureDialog.showModal();
  document.body.classList.add("dialog-open");
});
document.getElementById("close-figure").addEventListener("click", () => figureDialog.close());
figureDialog.addEventListener("click", event => {
  if (event.target !== figureDialog) return;
  const rect = figureDialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) figureDialog.close();
});
figureDialog.addEventListener("close", () => document.body.classList.remove("dialog-open"));

const demoSequence = "ATGCGTACGATCGATGCTAGCTAGCATCGATCGTACGTAGCTAGCTGA";
const demoPrefixLength = 4;
const demoNewBases = demoSequence.length - demoPrefixLength;
const demoDiffusionSteps = 8;
const demoRevealOrder = Array.from({ length: demoNewBases }, (_, index) => demoPrefixLength + (index * 17 + 9) % demoNewBases);
const generationDemo = document.getElementById("generation-demo");
const generationPlay = document.getElementById("generation-play");
const generationReset = document.getElementById("generation-reset");
const generationStatus = document.getElementById("generation-status");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let generationTick = 0;
let generationPreviousTick = 0;
let generationTimer = null;
let generationInteracted = false;

function createGenerationTokens(id) {
  const container = document.getElementById(id);
  return [...demoSequence].map(() => {
    const token = document.createElement("span");
    token.className = "generation-token";
    container.append(token);
    return token;
  });
}

const phagenTokens = createGenerationTokens("phagen-tokens");
const evoTokens = createGenerationTokens("evo-tokens");

function updateGenerationControls() {
  const finished = generationTick >= demoNewBases;
  const playing = generationTimer !== null;
  const playKey = finished ? "generationReplay" : playing ? "generationPause" : "generationPlay";
  generationPlay.textContent = translations[language][playKey];
  generationPlay.setAttribute("aria-pressed", String(playing));
  const statusKey = finished ? "generationComplete" : playing ? "generationPlaying" : generationTick > 0 ? "generationPaused" : "generationReady";
  const statusText = translations[language][statusKey];
  if (generationStatus.textContent !== statusText) generationStatus.textContent = statusText;
}

function renderGeneration() {
  const phagenStep = Math.min(generationTick, demoDiffusionSteps);
  const phagenCount = Math.ceil(demoNewBases * phagenStep / demoDiffusionSteps);
  const previousCount = Math.ceil(demoNewBases * Math.min(generationPreviousTick, demoDiffusionSteps) / demoDiffusionSteps);
  const revealed = new Set(demoRevealOrder.slice(0, phagenCount));
  const latest = new Set(demoRevealOrder.slice(previousCount, phagenCount));
  const evoCount = Math.min(generationTick, demoNewBases);
  phagenTokens.forEach((token, index) => {
    const prefix = index < demoPrefixLength;
    const generated = revealed.has(index);
    token.textContent = prefix || generated ? demoSequence[index] : "M";
    token.className = "generation-token " + (prefix ? "is-prefix" : generated ? "is-generated" : "is-masked");
    if (latest.has(index)) token.classList.add("is-latest");
  });
  evoTokens.forEach((token, index) => {
    const prefix = index < demoPrefixLength;
    const generated = index >= demoPrefixLength && index < demoPrefixLength + evoCount;
    token.textContent = prefix || generated ? demoSequence[index] : "";
    token.className = "generation-token " + (prefix ? "is-prefix" : generated ? "is-generated" : "is-empty");
    if (generated && generationTick !== generationPreviousTick && index === demoPrefixLength + evoCount - 1) token.classList.add("is-latest");
    if (index === demoPrefixLength + evoCount) token.classList.add("is-next");
  });
  document.getElementById("phagen-iterations").textContent = String(phagenStep);
  document.getElementById("phagen-new-bases").textContent = `${phagenCount} / ${demoNewBases}`;
  document.getElementById("evo-iterations").textContent = String(evoCount);
  document.getElementById("evo-new-bases").textContent = `${evoCount} / ${demoNewBases}`;
}

function pauseGeneration() {
  clearInterval(generationTimer);
  generationTimer = null;
  updateGenerationControls();
}

function resetGeneration() {
  clearInterval(generationTimer);
  generationTimer = null;
  generationTick = 0;
  generationPreviousTick = 0;
  renderGeneration();
  updateGenerationControls();
}

function playGeneration() {
  if (generationTimer !== null) return;
  if (generationTick >= demoNewBases) resetGeneration();
  generationTimer = setInterval(() => {
    generationPreviousTick = generationTick;
    generationTick = Math.min(generationTick + 1, demoNewBases);
    renderGeneration();
    if (generationTick >= demoNewBases) {
      clearInterval(generationTimer);
      generationTimer = null;
    }
    updateGenerationControls();
  }, 220);
  updateGenerationControls();
}

generationPlay.addEventListener("click", () => {
  generationInteracted = true;
  if (generationTimer === null) playGeneration();
  else pauseGeneration();
});
generationReset.addEventListener("click", () => {
  generationInteracted = true;
  resetGeneration();
});
reducedMotion.addEventListener("change", event => { if (event.matches) pauseGeneration(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) pauseGeneration(); });
renderGeneration();
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    observer.disconnect();
    if (!reducedMotion.matches && !generationInteracted) playGeneration();
  }, { threshold: 0.4 });
  observer.observe(generationDemo);
}

let storedLanguage;
try { storedLanguage = localStorage.getItem("phagen-language"); } catch (_) {}
setLanguage(storedLanguage || "en");
