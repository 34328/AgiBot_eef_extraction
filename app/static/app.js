const taskSelect = document.getElementById("taskSelect");
const episodeSelect = document.getElementById("episodeSelect");
const statusText = document.getElementById("statusText");
const countText = document.getElementById("countText");

const videoHead = document.getElementById("videoHead");
const videoLeft = document.getElementById("videoLeft");
const videoRight = document.getElementById("videoRight");
const masterVideo = videoHead;
let isSyncing = false;
let metaCache = {};
let isScrubbing = false;
let scrubRAF = null; // requestAnimationFrame ID for throttling scrub

// Task Info elements
const taskInfoPanel = document.getElementById("taskInfoPanel");
const taskNameText = document.getElementById("taskNameText");
const initSceneText = document.getElementById("initSceneText");
const actionsContainer = document.getElementById("actionsContainer");

// AI Score elements
const aiScoreBtn = document.getElementById("aiScoreBtn");
const aiScoreResult = document.getElementById("aiScoreResult");
const aiScoreValue = document.getElementById("aiScoreValue");

// Episode Navigation elements
const prevEpisodeBtn = document.getElementById("prevEpisodeBtn");
const nextEpisodeBtn = document.getElementById("nextEpisodeBtn");

const timeline = document.getElementById("timeline");
const timelineTrack = document.getElementById("timelineTrack");
const timelineRange = document.getElementById("timelineRange");
const timelineFill = document.getElementById("timelineFill");
const timelineThumb = document.getElementById("timelineThumb");
const timelineTooltip = document.getElementById("timelineTooltip");
const timelineStartHandle = document.getElementById("timelineStartHandle");
const timelineEndHandle = document.getElementById("timelineEndHandle");
const trimSaveBtn = document.getElementById("trimSaveBtn");
const viewEditToggle = document.getElementById("viewEditToggle");

let isEditMode = false;
let rangeStartRatio = 0;
let rangeEndRatio = 1;
let activeRangeHandle = null;
let pendingInitialNudge = false;
let saveFeedbackTimer = null;

const setStatus = (text, isLoading = false) => {
  statusText.textContent = text;
  statusText.style.color = isLoading ? "#4dd2ff" : "";
};

const clearVideos = () => {
  [videoHead, videoLeft, videoRight].forEach((v) => {
    v.pause();
    v.removeAttribute("src");
    v.load();
  });
};

const loadTasks = async () => {
  setStatus("Loading tasks...", true);
  const res = await fetch("/api/tasks");
  const data = await res.json();
  taskSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a task";
  taskSelect.appendChild(placeholder);

  data.tasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = task;
    option.textContent = task;
    taskSelect.appendChild(option);
  });
  countText.textContent = `${data.tasks.length} tasks available`;
  setStatus("Ready");
};

const loadEpisodes = async (task) => {
  setStatus("Loading episodes...", true);
  episodeSelect.disabled = true;
  episodeSelect.innerHTML = "";
  const res = await fetch(`/api/episodes/${task}`);
  const data = await res.json();

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select an episode";
  episodeSelect.appendChild(placeholder);

  data.episodes.forEach((ep) => {
    const option = document.createElement("option");
    option.value = ep;
    option.textContent = ep;
    episodeSelect.appendChild(option);
  });
  episodeSelect.disabled = false;
  countText.textContent = `${data.episodes.length} episodes in task ${task}`;
  setStatus("Ready");
};

// 当前加载的 episode 信息（用于重试机制）
let currentTask = null;
let currentEpisode = null;
let headRetryCount = 0;
const MAX_HEAD_RETRIES = 3;

const loadVideos = async (task, episode) => {
  setStatus("Loading videos...", true);
  currentTask = task;
  currentEpisode = episode;
  headRetryCount = 0;

  // 取消之前正在进行的视频生成任务
  try {
    await fetch("/api/cancel_video_generation", { method: "POST" });
  } catch (e) {
    console.warn("Failed to cancel video generation:", e);
  }

  clearVideos();
  metaCache = {};
  resetRangeSelection();
  setTrimSaveVisual("idle");

  loadHeadVideo(task, episode);
  videoLeft.src = `/api/video/${task}/${episode}/left`;
  videoRight.src = `/api/video/${task}/${episode}/right`;

  const checkLoaded = () => {
    const ready = [videoHead, videoLeft, videoRight].every(
      (v) => v.readyState >= 2
    );
    if (ready) {
      setStatus("Ready");
    }
  };

  [videoHead, videoLeft, videoRight].forEach((v) => {
    v.onloadeddata = checkLoaded;
  });

  await loadMeta(task, episode);
  await loadSavedRange(task, episode);
  pendingInitialNudge = isEditMode;
  if (isEditMode) {
    nudgePlayheadAfterRangeStart();
    if (masterVideo.duration) {
      pendingInitialNudge = false;
    }
  }
  loadTaskInfo(task, episode);

  // Reset AI score display and enable button
  resetAiScore();
  aiScoreBtn.disabled = false;

  // Update episode navigation buttons
  updateEpisodeNavButtons();
};

// 单独处理 head 视频加载，使用轮询机制
const loadHeadVideo = async (task, episode) => {
  // 如果 episode 已经切换，不加载
  if (task !== currentTask || episode !== currentEpisode) {
    return;
  }

  setStatus("Preparing EEF video...", true);

  // 触发后台生成
  try {
    const timestamp = new Date().getTime();
    const res = await fetch(`/api/prepare_video/${task}/${episode}?t=${timestamp}`);
    const data = await res.json();

    // 如果 episode 已切换，停止
    if (task !== currentTask || episode !== currentEpisode) {
      return;
    }

    if (data.status === "ready") {
      // 视频已就绪，直接加载，增加时间戳防止缓存问题
      const timestamp = new Date().getTime();
      videoHead.src = `/api/video/${task}/${episode}/head?t=${timestamp}`;
      setStatus("Ready");
      return;
    }
  } catch (e) {
    console.warn("Failed to prepare video:", e);
  }

  // 视频正在生成，开始轮询
  pollHeadVideo(task, episode, 0);
};

// 轮询检查 head 视频是否就绪
const MAX_POLL_ATTEMPTS = 240; // 最多轮询 120 秒 (2分钟)
const POLL_INTERVAL = 500; // 每 500ms 检查一次

const pollHeadVideo = async (task, episode, attempt) => {
  // 如果 episode 已切换，停止轮询
  if (task !== currentTask || episode !== currentEpisode) {
    return;
  }

  if (attempt >= MAX_POLL_ATTEMPTS) {
    setStatus("EEF video generation timeout");
    console.error("Head video polling timeout");
    return;
  }

  try {
    const timestamp = new Date().getTime();
    const res = await fetch(`/api/prepare_video/${task}/${episode}?t=${timestamp}`);

    // 再次检查 episode 是否切换
    if (task !== currentTask || episode !== currentEpisode) {
      return;
    }

    const data = await res.json();

    if (data.status === "ready") {
      // 视频就绪，加载，增加时间戳防止缓存问题
      const timestamp = new Date().getTime();
      videoHead.src = `/api/video/${task}/${episode}/head?t=${timestamp}`;
      setStatus("Ready");
      return;
    }

    if (data.status === "failed") {
      // 生成失败，显示错误信息
      const errorMsg = data.message || "视频生成失败";
      setStatus(`❌ ${errorMsg}`);
      alert(`⚠️ Head 视频生成失败\n\n${errorMsg}\n\n该 Episode 的数据可能存在问题，请检查数据完整性。`);
      return;
    }

    // 仍在生成，继续轮询
    const elapsed = Math.round(attempt * POLL_INTERVAL / 1000);
    setStatus(`Generating EEF video... ${elapsed}s`, true);
    setTimeout(() => pollHeadVideo(task, episode, attempt + 1), POLL_INTERVAL);
  } catch (e) {
    console.warn("Poll failed:", e);
    // 网络错误，等待后重试
    setTimeout(() => pollHeadVideo(task, episode, attempt + 1), POLL_INTERVAL);
  }
};

const loadTaskInfo = async (task, episode) => {
  try {
    const res = await fetch(`/api/task_info/${task}/${episode}`);
    if (!res.ok) {
      taskInfoPanel.style.display = "none";
      return;
    }
    const data = await res.json();

    // Display task name
    taskNameText.textContent = data.task_name || "Unknown Task";

    // Display init scene text
    initSceneText.textContent = data.init_scene_text || "";
    initSceneText.style.display = data.init_scene_text ? "block" : "none";

    // Display actions
    actionsContainer.innerHTML = "";
    if (data.actions && data.actions.length > 0) {
      data.actions.forEach((action) => {
        const actionItem = document.createElement("div");
        actionItem.className = "action-item";
        actionItem.innerHTML = `
          <span class="action-frames">Frame ${action.start_frame} → ${action.end_frame}</span>
          <span class="action-text">${action.action_text}</span>
          <span class="action-skill">${action.skill}</span>
        `;
        actionsContainer.appendChild(actionItem);
      });
    }

    taskInfoPanel.style.display = "flex";
  } catch {
    taskInfoPanel.style.display = "none";
  }
};

const hideTaskInfo = () => {
  taskInfoPanel.style.display = "none";
};

// AI Score functions
const resetAiScore = () => {
  aiScoreBtn.disabled = true;
  aiScoreBtn.classList.remove("loading");
  aiScoreResult.style.display = "none";
  aiScoreValue.textContent = "--";
};

const requestAiScore = async () => {
  const task = taskSelect.value;
  const episode = episodeSelect.value;
  if (!task || !episode) return;

  // Set loading state
  aiScoreBtn.classList.add("loading");
  aiScoreBtn.querySelector(".ai-score-btn-text").textContent = "Scoring...";
  aiScoreResult.style.display = "none";

  try {
    const res = await fetch(`/api/score/${task}/${episode}`, { method: "POST" });
    const data = await res.json();

    if (res.ok && data.score !== undefined) {
      aiScoreValue.textContent = data.score;
      aiScoreResult.style.display = "flex";
    } else {
      aiScoreValue.textContent = "Error";
      aiScoreResult.style.display = "flex";
      console.error("AI Score error:", data.error);
    }
  } catch (err) {
    aiScoreValue.textContent = "Failed";
    aiScoreResult.style.display = "flex";
    console.error("AI Score request failed:", err);
  } finally {
    aiScoreBtn.classList.remove("loading");
    aiScoreBtn.querySelector(".ai-score-btn-text").textContent = "AI Score";
  }
};

const getTimelineInfo = () => {
  const duration = masterVideo.duration || 0;
  const frames = metaCache?.head?.frames;
  let fps = metaCache?.head?.fps;
  if (!fps && frames && duration) {
    fps = frames / duration;
  }
  return { duration, frames, fps };
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const getTotalFrames = () => {
  const { duration, frames, fps } = getTimelineInfo();
  if (Number.isInteger(frames) && frames > 1) return frames;
  if (duration > 0 && fps) {
    const estimated = Math.round(duration * fps);
    if (estimated > 1) return estimated;
  }
  return 1;
};

const frameToRatio = (frame) => {
  const total = getTotalFrames();
  if (total <= 1) return 0;
  return clamp(frame / (total - 1), 0, 1);
};

const ratioToFrame = (ratio) => {
  const total = getTotalFrames();
  if (total <= 1) return 0;
  return Math.round(clamp(ratio, 0, 1) * (total - 1));
};

const getActiveRange = () => {
  if (!isEditMode) {
    return { startRatio: 0, endRatio: 1 };
  }
  return { startRatio: rangeStartRatio, endRatio: rangeEndRatio };
};

const getCurrentRangeFrames = () => ({
  startFrame: ratioToFrame(rangeStartRatio),
  endFrame: ratioToFrame(rangeEndRatio),
  totalFrames: getTotalFrames(),
});

const resetRangeSelection = () => {
  rangeStartRatio = 0;
  rangeEndRatio = 1;
  renderTimelineRange();
};

const setTrimSaveVisual = (state) => {
  if (!trimSaveBtn) return;
  if (saveFeedbackTimer) {
    clearTimeout(saveFeedbackTimer);
    saveFeedbackTimer = null;
  }

  trimSaveBtn.classList.remove("is-saving", "is-saved", "is-error");
  trimSaveBtn.disabled = false;

  if (state === "saving") {
    trimSaveBtn.classList.add("is-saving");
    trimSaveBtn.textContent = "Saving...";
    trimSaveBtn.disabled = true;
    return;
  }

  if (state === "saved") {
    trimSaveBtn.classList.add("is-saved");
    trimSaveBtn.textContent = "Saved";
    saveFeedbackTimer = setTimeout(() => setTrimSaveVisual("idle"), 1200);
    return;
  }

  if (state === "cleared") {
    trimSaveBtn.classList.add("is-saved");
    trimSaveBtn.textContent = "Cleared";
    saveFeedbackTimer = setTimeout(() => setTrimSaveVisual("idle"), 1200);
    return;
  }

  if (state === "error") {
    trimSaveBtn.classList.add("is-error");
    trimSaveBtn.textContent = "Retry";
    saveFeedbackTimer = setTimeout(() => setTrimSaveVisual("idle"), 1800);
    return;
  }

  trimSaveBtn.textContent = "Save";
};

const renderTimelineRange = () => {
  if (!timelineRange || !timelineStartHandle || !timelineEndHandle) return;
  const startPercent = rangeStartRatio * 100;
  const endPercent = rangeEndRatio * 100;
  timelineRange.style.left = `${startPercent}%`;
  timelineRange.style.width = `${Math.max(0, endPercent - startPercent)}%`;
  timelineStartHandle.style.left = `${startPercent}%`;
  timelineEndHandle.style.left = `${endPercent}%`;
};

const setAllVideoTime = (time) => {
  [videoHead, videoLeft, videoRight].forEach((v) => {
    if (v.readyState < 1) return;
    if (v.fastSeek) {
      v.fastSeek(time);
    } else {
      v.currentTime = time;
    }
  });
};

const updateTimeline = () => {
  const { duration } = getTimelineInfo();
  if (!duration) {
    timelineFill.style.left = "0%";
    timelineFill.style.width = "0%";
    timelineThumb.style.left = "0%";
    timelineThumb.classList.remove("is-overlap-start", "is-overlap-end");
    return;
  }
  const rawProgress = clamp(masterVideo.currentTime / duration, 0, 1);
  if (!isEditMode) {
    const percent = rawProgress * 100;
    timelineFill.style.left = "0%";
    timelineFill.style.width = `${percent}%`;
    timelineThumb.style.left = `${percent}%`;
    timelineThumb.classList.remove("is-overlap-start", "is-overlap-end");
    return;
  }

  const startRatio = rangeStartRatio;
  const endRatio = rangeEndRatio;
  const bounded = clamp(rawProgress, startRatio, endRatio);
  const startPercent = startRatio * 100;
  const endPercent = endRatio * 100;
  const thumbPercent = bounded * 100;

  timelineFill.style.left = `${startPercent}%`;
  timelineFill.style.width = `${Math.max(0, thumbPercent - startPercent)}%`;
  timelineThumb.style.left = `${thumbPercent}%`;

  const trackWidth = timelineTrack?.getBoundingClientRect().width || 0;
  const gapPercent = trackWidth > 0 ? (10 / trackWidth) * 100 : 0.8;
  const nearStart = Math.abs(thumbPercent - startPercent) <= gapPercent;
  const nearEnd = Math.abs(thumbPercent - endPercent) <= gapPercent;
  timelineThumb.classList.toggle("is-overlap-start", nearStart && !nearEnd);
  timelineThumb.classList.toggle("is-overlap-end", nearEnd && !nearStart);
};

const updateTooltip = (clientX) => {
  if (!timelineTrack) return;
  const rect = timelineTrack.getBoundingClientRect();
  const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
  const ratioOnUi = rect.width ? x / rect.width : 0;
  const { startRatio, endRatio } = getActiveRange();
  const ratio = isEditMode ? clamp(ratioOnUi, startRatio, endRatio) : ratioOnUi;
  const { duration, frames, fps } = getTimelineInfo();
  let currentFrame = 0;
  let totalFrames = 0;
  if (frames) {
    totalFrames = frames;
    currentFrame = ratioToFrame(ratio);
  } else if (duration && fps) {
    totalFrames = Math.round(duration * fps);
    currentFrame = ratioToFrame(ratio);
  }
  timelineTooltip.textContent = `${currentFrame}/${totalFrames}`;
  timelineTooltip.style.left = `${x}px`;
};

const seekWithRatio = (ratioOnUi) => {
  const { duration } = getTimelineInfo();
  if (!duration) return;
  const ratio = clamp(ratioOnUi, 0, 1);
  const { startRatio, endRatio } = getActiveRange();
  const effectiveRatio = isEditMode ? clamp(ratio, startRatio, endRatio) : ratio;
  const targetTime = effectiveRatio * duration;

  // Use fastSeek when scrubbing for better performance (if available)
  if (isScrubbing && masterVideo.fastSeek) {
    masterVideo.fastSeek(targetTime);
  } else {
    masterVideo.currentTime = targetTime;
  }

  // Also sync follower videos during scrub
  if (isScrubbing) {
    [videoLeft, videoRight].forEach((v) => {
      if (v.fastSeek) {
        v.fastSeek(targetTime);
      } else {
        v.currentTime = targetTime;
      }
    });
  }
};

const enforceEditPlaybackRange = () => {
  if (!isEditMode) return;
  const { duration } = getTimelineInfo();
  if (!duration) return;
  const startTime = duration * rangeStartRatio;
  const endTime = duration * rangeEndRatio;
  const epsilon = 1 / 120;
  if (masterVideo.currentTime < startTime) {
    setAllVideoTime(startTime);
    return;
  }
  if (masterVideo.currentTime > endTime - epsilon) {
    setAllVideoTime(endTime);
    [videoHead, videoLeft, videoRight].forEach((v) => v.pause());
  }
};

const nudgePlayheadAfterRangeStart = () => {
  const { duration } = getTimelineInfo();
  if (!duration) return;
  const total = getTotalFrames();
  const startFrame = ratioToFrame(rangeStartRatio);
  const targetFrame = total > 1 ? Math.min(startFrame + 1, total - 1) : startFrame;
  const targetRatio = frameToRatio(targetFrame);
  const targetTime = targetRatio * duration;
  setAllVideoTime(targetTime);
  updateTimeline();
};

const applyFrameRange = (startFrame, endFrame) => {
  const total = getTotalFrames();
  const safeStart = clamp(startFrame, 0, Math.max(0, total - 1));
  const safeEnd = clamp(endFrame, safeStart, Math.max(0, total - 1));
  rangeStartRatio = frameToRatio(safeStart);
  rangeEndRatio = frameToRatio(safeEnd);
  renderTimelineRange();
};

const bindTimelineEvents = () => {
  if (!timeline || !timelineTrack || !timelineThumb) return;

  const performScrub = (clientX) => {
    const rect = timelineTrack.getBoundingClientRect();
    seekWithRatio((clientX - rect.left) / rect.width);
  };

  timelineTrack.addEventListener("mousemove", (event) => {
    updateTooltip(event.clientX);
    if (isScrubbing) {
      // Use requestAnimationFrame to throttle scrub updates
      if (scrubRAF) cancelAnimationFrame(scrubRAF);
      scrubRAF = requestAnimationFrame(() => {
        performScrub(event.clientX);
      });
    }
  });

  timelineTrack.addEventListener("mousedown", (event) => {
    if (activeRangeHandle) return;
    isScrubbing = true;
    // Pause videos during scrub for smoother experience
    [videoHead, videoLeft, videoRight].forEach(v => v.pause());
    performScrub(event.clientX);
  });

  timelineThumb.addEventListener("mousedown", (event) => {
    if (!timelineThumb || activeRangeHandle) return;
    event.preventDefault();
    event.stopPropagation();
    isScrubbing = true;
    [videoHead, videoLeft, videoRight].forEach(v => v.pause());
    performScrub(event.clientX);
  });

  window.addEventListener("mousemove", (event) => {
    if (!isScrubbing) return;
    if (scrubRAF) cancelAnimationFrame(scrubRAF);
    scrubRAF = requestAnimationFrame(() => {
      performScrub(event.clientX);
      updateTooltip(event.clientX);
    });
  });

  window.addEventListener("mouseup", () => {
    if (isScrubbing) {
      isScrubbing = false;
      if (scrubRAF) {
        cancelAnimationFrame(scrubRAF);
        scrubRAF = null;
      }
    }
    activeRangeHandle = null;
  });
};

const syncCurrentTime = async (time) => {
  const followers = [videoLeft, videoRight];
  followers.forEach((video) => {
    if (video.readyState >= 1) {
      // Use fastSeek when available for better performance
      if (video.fastSeek) {
        video.fastSeek(time);
      } else {
        video.currentTime = time;
      }
    }
  });
};

const bindSyncEvents = () => {
  masterVideo.addEventListener("play", async () => {
    if (isSyncing) return;
    if (isEditMode) {
      const { duration } = getTimelineInfo();
      if (duration > 0) {
        const startTime = duration * rangeStartRatio;
        const endTime = duration * rangeEndRatio;
        if (masterVideo.currentTime < startTime || masterVideo.currentTime > endTime) {
          setAllVideoTime(startTime);
        }
      }
    }
    isSyncing = true;
    try {
      await syncCurrentTime(masterVideo.currentTime);
      await Promise.all(
        [videoLeft, videoRight].map((v) => v.play().catch(() => null))
      );
    } finally {
      isSyncing = false;
    }
  });

  masterVideo.addEventListener("pause", () => {
    if (isSyncing) return;
    isSyncing = true;
    [videoLeft, videoRight].forEach((v) => v.pause());
    isSyncing = false;
  });

  masterVideo.addEventListener("seeking", async () => {
    if (isSyncing) return;
    isSyncing = true;
    try {
      await syncCurrentTime(masterVideo.currentTime);
      enforceEditPlaybackRange();
    } finally {
      isSyncing = false;
    }
  });

  masterVideo.addEventListener("timeupdate", () => {
    enforceEditPlaybackRange();
    updateTimeline();
  });
  masterVideo.addEventListener("loadedmetadata", () => {
    updateTimeline();
    if (pendingInitialNudge && isEditMode) {
      nudgePlayheadAfterRangeStart();
      pendingInitialNudge = false;
    }
  });
};

const loadMeta = async (task, episode) => {
  try {
    const res = await fetch(`/api/meta/${task}/${episode}`);
    const data = await res.json();
    metaCache = data || {};
    const head = data.head?.frames;
    const left = data.left?.frames;
    const right = data.right?.frames;
    if (head == null || left == null || right == null) {
      setStatus("Frame info unavailable");
      return data;
    }
    const allEqual = head === left && head === right;
    setStatus(allEqual ? "Frames OK" : "Frames mismatch");
    return data;
  } catch {
    setStatus("Frame info unavailable");
    return null;
  }
};

const loadSavedRange = async (task, episode) => {
  try {
    const res = await fetch(`/api/filter/${task}/${episode}`);
    if (!res.ok) {
      resetRangeSelection();
      renderTimelineRange();
      return;
    }
    const data = await res.json();

    if (task !== currentTask || episode !== currentEpisode) {
      return;
    }

    const saved = data?.range;
    if (
      saved &&
      Number.isInteger(saved.start_frame) &&
      Number.isInteger(saved.end_frame) &&
      saved.end_frame >= saved.start_frame
    ) {
      applyFrameRange(saved.start_frame, saved.end_frame);
    } else {
      resetRangeSelection();
    }
    renderTimelineRange();
  } catch {
    resetRangeSelection();
    renderTimelineRange();
  }
};

const updateRangeFromPointer = (clientX) => {
  if (!activeRangeHandle || !timelineTrack) return;
  const rect = timelineTrack.getBoundingClientRect();
  const ratio = rect.width ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0;
  const minGap = 1 / Math.max(1, getTotalFrames() - 1);
  if (activeRangeHandle === "start") {
    rangeStartRatio = clamp(Math.min(ratio, rangeEndRatio - minGap), 0, 1);
  } else {
    rangeEndRatio = clamp(Math.max(ratio, rangeStartRatio + minGap), 0, 1);
  }
  renderTimelineRange();
  enforceEditPlaybackRange();
  updateTimeline();
};

const bindRangeHandleEvents = () => {
  if (!timelineStartHandle || !timelineEndHandle) return;

  timelineStartHandle.addEventListener("mousedown", (event) => {
    if (!isEditMode) return;
    event.preventDefault();
    event.stopPropagation();
    activeRangeHandle = "start";
  });

  timelineEndHandle.addEventListener("mousedown", (event) => {
    if (!isEditMode) return;
    event.preventDefault();
    event.stopPropagation();
    activeRangeHandle = "end";
  });

  window.addEventListener("mousemove", (event) => {
    if (!activeRangeHandle) return;
    updateRangeFromPointer(event.clientX);
  });
};

const saveRangeSelection = async () => {
  const task = taskSelect.value;
  const episode = episodeSelect.value;
  if (!task || !episode || !isEditMode) return;
  const { startFrame, endFrame, totalFrames } = getCurrentRangeFrames();

  try {
    setTrimSaveVisual("saving");
    const res = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task,
        episode,
        start_frame: startFrame,
        end_frame: endFrame,
        total_frames: totalFrames,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus(data.error || "Save failed");
      setTrimSaveVisual("error");
      return;
    }
    setStatus(data.saved ? "Range saved" : "Full range cleared");
    setTrimSaveVisual(data.saved ? "saved" : "cleared");
  } catch {
    setStatus("Save failed");
    setTrimSaveVisual("error");
  }
};

const setModeState = (isEdit) => {
  if (!viewEditToggle) return;
  const wasEdit = isEditMode;
  isEditMode = isEdit;
  viewEditToggle.classList.toggle("is-edit", isEdit);
  viewEditToggle.setAttribute("aria-pressed", isEdit ? "true" : "false");
  document.body.dataset.mode = isEdit ? "edit" : "view";
  renderTimelineRange();
  updateTimeline();
  if (isEdit && !wasEdit) {
    nudgePlayheadAfterRangeStart();
    pendingInitialNudge = false;
    setTrimSaveVisual("idle");
  } else if (!isEdit) {
    pendingInitialNudge = false;
    setTrimSaveVisual("idle");
  }
};

const bindModeToggle = () => {
  if (!viewEditToggle) return;
  setModeState(false);
  viewEditToggle.addEventListener("click", () => {
    const isEdit = viewEditToggle.getAttribute("aria-pressed") !== "true";
    setModeState(isEdit);
  });
};

taskSelect.addEventListener("change", () => {
  const task = taskSelect.value;
  clearVideos();
  hideTaskInfo();
  if (!task) {
    episodeSelect.disabled = true;
    episodeSelect.innerHTML = "<option value=\"\">Select a task first</option>";
    countText.textContent = "";
    return;
  }
  loadEpisodes(task);
});

episodeSelect.addEventListener("change", () => {
  const task = taskSelect.value;
  const episode = episodeSelect.value;
  if (!task || !episode) {
    return;
  }
  loadVideos(task, episode);
});

bindSyncEvents();
bindTimelineEvents();
bindRangeHandleEvents();
bindModeToggle();

// AI Score button event
aiScoreBtn.addEventListener("click", requestAiScore);
trimSaveBtn.addEventListener("click", saveRangeSelection);

// Episode Navigation functions
const updateEpisodeNavButtons = () => {
  const options = episodeSelect.options;
  const currentIndex = episodeSelect.selectedIndex;

  // Enable/disable buttons based on position
  prevEpisodeBtn.disabled = currentIndex <= 1; // index 0 is placeholder
  nextEpisodeBtn.disabled = currentIndex >= options.length - 1 || currentIndex < 1;
};

const goToPrevEpisode = () => {
  const currentIndex = episodeSelect.selectedIndex;
  if (currentIndex <= 1) {
    alert("已经是第一个 Episode 了，没有上一个了！");
    return;
  }
  episodeSelect.selectedIndex = currentIndex - 1;
  episodeSelect.dispatchEvent(new Event("change"));
};

const goToNextEpisode = () => {
  const options = episodeSelect.options;
  const currentIndex = episodeSelect.selectedIndex;
  if (currentIndex >= options.length - 1) {
    alert("已经是最后一个 Episode 了，没有下一个了！");
    return;
  }
  episodeSelect.selectedIndex = currentIndex + 1;
  episodeSelect.dispatchEvent(new Event("change"));
};

prevEpisodeBtn.addEventListener("click", goToPrevEpisode);
nextEpisodeBtn.addEventListener("click", goToNextEpisode);

loadTasks().catch(() => {
  setStatus("Failed to load tasks");
});
