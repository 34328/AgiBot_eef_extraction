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

// Task Info elements
const taskInfoPanel = document.getElementById("taskInfoPanel");
const taskNameText = document.getElementById("taskNameText");
const initSceneText = document.getElementById("initSceneText");
const actionsContainer = document.getElementById("actionsContainer");

const timeline = document.getElementById("timeline");
const timelineFill = document.getElementById("timelineFill");
const timelineThumb = document.getElementById("timelineThumb");
const timelineTooltip = document.getElementById("timelineTooltip");

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

const loadVideos = (task, episode) => {
  setStatus("Loading videos...", true);
  clearVideos();
  metaCache = {};

  videoHead.src = `/api/video/${task}/${episode}/head`;
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

  loadMeta(task, episode);
  loadTaskInfo(task, episode);
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

const getTimelineInfo = () => {
  const duration = masterVideo.duration || 0;
  const frames = metaCache?.head?.frames;
  let fps = metaCache?.head?.fps;
  if (!fps && frames && duration) {
    fps = frames / duration;
  }
  return { duration, frames, fps };
};

const updateTimeline = () => {
  const { duration } = getTimelineInfo();
  if (!duration) {
    timelineFill.style.width = "0%";
    timelineThumb.style.left = "0%";
    return;
  }
  const progress = masterVideo.currentTime / duration;
  const percent = Math.max(0, Math.min(1, progress)) * 100;
  timelineFill.style.width = `${percent}%`;
  timelineThumb.style.left = `${percent}%`;
};

const updateTooltip = (clientX) => {
  const rect = timeline.getBoundingClientRect();
  const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
  const ratio = rect.width ? x / rect.width : 0;
  const { duration, frames, fps } = getTimelineInfo();
  let currentFrame = 0;
  let totalFrames = 0;
  if (frames) {
    totalFrames = frames;
    currentFrame = Math.round(frames * ratio);
  } else if (duration && fps) {
    totalFrames = Math.round(duration * fps);
    currentFrame = Math.round(totalFrames * ratio);
  }
  timelineTooltip.textContent = `${currentFrame}/${totalFrames}`;
  timelineTooltip.style.left = `${x}px`;
};

const seekWithRatio = (ratio) => {
  const { duration } = getTimelineInfo();
  if (!duration) return;
  masterVideo.currentTime = Math.max(0, Math.min(1, ratio)) * duration;
};

const bindTimelineEvents = () => {
  if (!timeline) return;
  timeline.addEventListener("mousemove", (event) => {
    updateTooltip(event.clientX);
    if (isScrubbing) {
      const rect = timeline.getBoundingClientRect();
      seekWithRatio((event.clientX - rect.left) / rect.width);
    }
  });

  timeline.addEventListener("mousedown", (event) => {
    isScrubbing = true;
    const rect = timeline.getBoundingClientRect();
    seekWithRatio((event.clientX - rect.left) / rect.width);
  });

  window.addEventListener("mouseup", () => {
    isScrubbing = false;
  });
};

const syncCurrentTime = async (time) => {
  const followers = [videoLeft, videoRight];
  await Promise.all(
    followers.map(
      (video) =>
        new Promise((resolve) => {
          if (video.readyState >= 1) {
            video.currentTime = time;
            resolve();
          } else {
            const onMeta = () => {
              video.removeEventListener("loadedmetadata", onMeta);
              video.currentTime = time;
              resolve();
            };
            video.addEventListener("loadedmetadata", onMeta);
          }
        })
    )
  );
};

const bindSyncEvents = () => {
  masterVideo.addEventListener("play", async () => {
    if (isSyncing) return;
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
    } finally {
      isSyncing = false;
    }
  });

  masterVideo.addEventListener("timeupdate", updateTimeline);
  masterVideo.addEventListener("loadedmetadata", updateTimeline);
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
      return;
    }
    const allEqual = head === left && head === right;
    setStatus(allEqual ? "Frames OK" : "Frames mismatch");
  } catch {
    setStatus("Frame info unavailable");
  }
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

loadTasks().catch(() => {
  setStatus("Failed to load tasks");
});
