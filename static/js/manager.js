(function () {
  'use strict';

  var legacyDismissedVisionStorageKey = 'managerDismissedVisionJobs';
  var state = { csrf: '', map: null, eventLayers: null, visionLayers: null, preview: null, anchorMarker: null, pickFeedbackLayers: [], pickRequestId: 0, picked: null, anchor: null, sourceJobId: null, sourceCandidateIndex: null, reviewDrafts: {}, legacyDismissedVisionJobs: loadLegacyDismissedVisionJobs(), deletedVisionJobs: new Set(), visionDeleteBusy: new Set(), legacyDeleteBusy: false, picking: false, pickPurpose: null, selectedRegionForRoad: null, maxImageBytes: 24 * 1024 * 1024, maxVideoBytes: 1024 * 1024 * 1024, uploadChunkBytes: 8 * 1024 * 1024, inferenceReady: false, visionCapabilities: {}, visionStatusBusy: false, visionUploadBusy: false, mediaPreviewUrl: '', mediaInvalidReason: '', impactPreviewRequestId: 0, observationRegions: [], observationScenes: [], observationSceneId: null, frameSignature: null, currentRegionPoints: [], drawingRegion: false, mediaObjectUrl: null, poll: null };
  var pendingVisionUploadKey = 'whu-walker-pending-vision-video-v1';
  var byId = function (id) { return document.getElementById(id); };
  var message = function (id, text, good) {
    var node = byId(id);
    if (!node) return;
    node.textContent = text || '';
    node.style.color = good ? '#397653' : '';
  };

  function loadLegacyDismissedVisionJobs() {
    try {
      var stored = JSON.parse(window.sessionStorage.getItem(legacyDismissedVisionStorageKey) || '[]');
      return new Set(Array.isArray(stored) ? stored.filter(function (id) { return typeof id === 'string'; }) : []);
    } catch (_) {
      return new Set();
    }
  }

  function persistLegacyDismissedVisionJobs() {
    try {
      if (state.legacyDismissedVisionJobs.size) {
        window.sessionStorage.setItem(legacyDismissedVisionStorageKey, JSON.stringify(Array.from(state.legacyDismissedVisionJobs)));
      } else {
        window.sessionStorage.removeItem(legacyDismissedVisionStorageKey);
      }
    } catch (_) {}
  }

  function isVisionJobHidden(jobId) {
    return state.deletedVisionJobs.has(jobId) || state.legacyDismissedVisionJobs.has(jobId);
  }

  async function migrateLegacyDismissedVisionJobs() {
    if (state.legacyDeleteBusy || !state.legacyDismissedVisionJobs.size) return;
    state.legacyDeleteBusy = true;
    var legacyIds = Array.from(state.legacyDismissedVisionJobs);
    for (var i = 0; i < legacyIds.length; i += 1) {
      var jobId = legacyIds[i];
      try {
        await request('/api/manager/vision-jobs/' + encodeURIComponent(jobId), 'DELETE');
        state.legacyDismissedVisionJobs.delete(jobId);
      } catch (_) {
        // Keep the old local hide marker and retry on the next manager poll.
      }
    }
    persistLegacyDismissedVisionJobs();
    state.legacyDeleteBusy = false;
  }

  function retryVisionSelection() {
    var uploadCard = byId('vision-upload-card');
    if (uploadCard && uploadCard.scrollIntoView) uploadCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    byId('media-file').click();
  }

  async function deleteVisionJob(jobId) {
    if (state.visionDeleteBusy.has(jobId)) return;
    if (!window.confirm('永久删除这条影像记录及上传文件？删除后无法恢复。')) return;
    state.visionDeleteBusy.add(jobId);
    try {
      await request('/api/manager/vision-jobs/' + encodeURIComponent(jobId), 'DELETE');
      state.deletedVisionJobs.add(jobId);
      state.legacyDismissedVisionJobs.delete(jobId);
      persistLegacyDismissedVisionJobs();
      refreshVisionJobs(true);
    } catch (error) {
      window.alert(error.message || '影像记录删除失败，请重试。');
    } finally {
      state.visionDeleteBusy.delete(jobId);
    }
  }

  function isVideoFile(file) {
    return !!file && (/^video\//i.test(file.type || '') || /\.(mp4|mov|avi|webm)$/i.test(file.name || ''));
  }

  function isImageFile(file) {
    return !!file && (/^image\/(jpeg|png|webp)$/i.test(file.type || '') || /\.(jpe?g|png|webp)$/i.test(file.name || ''));
  }

  function formatBytes(value) {
    if (value < 1024) return value + ' B';
    if (value < 1024 * 1024) return (value / 1024).toFixed(1) + ' KB';
    return (value / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function clearMediaPreview() {
    if (state.mediaPreviewUrl) URL.revokeObjectURL(state.mediaPreviewUrl);
    state.mediaPreviewUrl = '';
    var preview = byId('selected-media-preview');
    preview.removeAttribute('src');
    preview.hidden = true;
  }

  function clearSelectedMedia() {
    clearMediaPreview();
    state.mediaInvalidReason = '';
    byId('media-file').value = '';
    byId('selected-media').hidden = true;
    byId('selected-media-name').textContent = '';
    byId('selected-media-meta').textContent = '';
  }

  function renderSelectedMedia(file) {
    clearMediaPreview();
    state.mediaInvalidReason = '';
    if (!file) {
      byId('selected-media').hidden = true;
      return;
    }
    var mediaKind = isImageFile(file) ? '图片' : isVideoFile(file) ? '视频' : '不支持的文件';
    if (mediaKind === '不支持的文件') {
      state.mediaInvalidReason = '文件格式不受支持，请选择 JPG、PNG、WebP 图片或 MP4、MOV、AVI、WebM 视频。';
    } else {
      var limit = isVideoFile(file) ? state.maxVideoBytes : state.maxImageBytes;
      if (file.size > limit) {
        state.mediaInvalidReason = '文件大小为 ' + formatBytes(file.size) + '，超过当前 ' + formatBytes(limit) + ' 上限。';
      }
    }
    byId('selected-media-name').textContent = file.name || '未命名影像';
    byId('selected-media-meta').textContent = formatBytes(file.size || 0) + ' · ' + mediaKind;
    byId('selected-media').hidden = false;
    if (isImageFile(file)) {
      var preview = byId('selected-media-preview');
      state.mediaPreviewUrl = URL.createObjectURL(file);
      preview.src = state.mediaPreviewUrl;
      preview.hidden = false;
      preview.onerror = function () {
        preview.hidden = true;
        if (!state.mediaInvalidReason) {
          message('vision-message', '图片已选择，但无法生成本地预览；仍可尝试识别。');
        }
      };
    }
  }

  function updateVisionGuidance() {
    var file = byId('media-file').files[0];
    if (!state.inferenceReady) {
      message('vision-message', file ? '视觉服务暂未就绪，已保留所选文件，请等待服务恢复。' : '视觉服务暂未就绪，请等待服务恢复。');
    } else if (!file) {
      message('vision-message', '先选择一张图片或一段短视频。');
    } else if (state.mediaInvalidReason) {
      message('vision-message', state.mediaInvalidReason);
    } else if (isVideoFile(file) && fileMatchesPending(file, readPendingVisionUpload())) {
      message('vision-message', '发现这段视频的未完成上传；提交后会从已上传位置继续。', true);
    } else {
      message('vision-message', '影像已准备好，可以开始识别；如有有效候选，将在复核后选择具体道路。', true);
    }
  }

  function canSubmitVision() {
    return !!(state.inferenceReady && !state.visionUploadBusy && !state.mediaInvalidReason && byId('media-file').files.length);
  }

  function updateVisionSubmitState() {
    byId('submit-vision').disabled = !canSubmitVision();
  }

  function regionKindLabel(kind) {
    return { vehicle_lane: '机动车道', pedestrian: '人行区域', road_surface: '路面积水检查', parking: '停车区域', exclude: '忽略区域' }[kind] || '观察区域';
  }

  function drawObservationRegions() {
    var canvas = byId('observation-canvas');
    if (!canvas || !canvas.width || !canvas.height || !canvas.getContext) return;
    var context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    function drawPolygon(points, color, close) {
      if (!points.length) return;
      context.beginPath();
      context.moveTo(points[0][0] * canvas.width, points[0][1] * canvas.height);
      points.slice(1).forEach(function (point) { context.lineTo(point[0] * canvas.width, point[1] * canvas.height); });
      if (close && points.length >= 3) context.closePath();
      context.strokeStyle = color; context.lineWidth = Math.max(2, canvas.width / 400);
      context.fillStyle = color.replace('1)', '0.16)');
      if (close && points.length >= 3) context.fill();
      context.stroke();
      points.forEach(function (point) {
        context.beginPath(); context.arc(point[0] * canvas.width, point[1] * canvas.height, Math.max(3, canvas.width / 250), 0, Math.PI * 2);
        context.fillStyle = '#fff'; context.fill(); context.stroke();
      });
    }
    state.observationRegions.forEach(function (region, index) {
      var colors = ['rgba(73,146,111,1)', 'rgba(223,151,62,1)', 'rgba(87,133,180,1)', 'rgba(186,84,84,1)'];
      drawPolygon(region.polygon, colors[index % colors.length], true);
    });
    if (state.currentRegionPoints.length) drawPolygon(state.currentRegionPoints, 'rgba(244,190,78,1)', false);
  }

  function setObservationCanvasSize() {
    var canvas = byId('observation-canvas');
    var image = byId('observation-image');
    var video = byId('observation-video');
    var width = video && !video.hidden && video.videoWidth ? video.videoWidth : image && image.naturalWidth;
    var height = video && !video.hidden && video.videoHeight ? video.videoHeight : image && image.naturalHeight;
    if (!canvas || !width || !height) return;
    var preview = byId('observation-editor');
    var stage = document.querySelector ? document.querySelector('.observation-preview') : null;
    var scale = Math.min(1, 8192 / width, 8192 / height);
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    if (stage && stage.style) stage.style.aspectRatio = width + ' / ' + height;
    if (preview) preview.hidden = false;
    drawObservationRegions();
  }

  function showObservationMedia(file) {
    var editor = byId('observation-editor');
    if (!editor) return;
    if (state.mediaObjectUrl && window.URL && window.URL.revokeObjectURL) window.URL.revokeObjectURL(state.mediaObjectUrl);
    state.mediaObjectUrl = null;
    editor.hidden = true;
    var image = byId('observation-image');
    var video = byId('observation-video');
    var canvas = byId('observation-canvas');
    if (image) { image.hidden = true; image.src = ''; }
    if (video) {
      if (video.pause) video.pause();
      video.hidden = true; video.src = '';
      if (video.load) video.load();
    }
    if (!file || !window.URL || !window.URL.createObjectURL) return;
    state.mediaObjectUrl = window.URL.createObjectURL(file);
    if (isVideoFile(file)) {
      video.hidden = false; video.src = state.mediaObjectUrl;
      video.addEventListener('loadedmetadata', setObservationCanvasSize, { once: true });
      video.addEventListener('loadeddata', function () {
        setObservationCanvasSize();
        try { state.frameSignature = observationFrameSignature(); } catch (_) {}
      }, { once: true });
      video.addEventListener('seeked', function () {
        drawObservationRegions();
        try { state.frameSignature = observationFrameSignature(); } catch (_) {}
      });
    } else {
      image.hidden = false; image.src = state.mediaObjectUrl;
      image.addEventListener('load', function () {
        setObservationCanvasSize();
        try { state.frameSignature = observationFrameSignature(); } catch (_) {}
      }, { once: true });
    }
    if (canvas) { canvas.width = 0; canvas.height = 0; }
    editor.hidden = false;
  }

  function renderObservationRegionList() {
    var root = byId('observation-region-list');
    if (!root) return;
    root.replaceChildren();
    if (!state.observationRegions.length) {
      root.textContent = '尚未圈选区域';
      return;
    }
    state.observationRegions.forEach(function (region, index) {
      var chip = document.createElement('span'); chip.className = 'observation-region-chip';
      chip.appendChild(document.createTextNode(regionKindLabel(region.kind) + ' ' + (index + 1)));
      if (region.road_link && region.road_link.road_name) {
        var roadName = document.createElement('span'); roadName.className = 'region-road-name';
        roadName.textContent = region.road_link.road_name;
        chip.appendChild(roadName);
      }
      if (region.kind !== 'exclude') {
        var bind = document.createElement('button'); bind.type = 'button'; bind.className = 'region-bind-road';
        bind.textContent = region.road_link ? '重选道路' : '绑定道路';
        bind.setAttribute('aria-label', '为' + regionKindLabel(region.kind) + '绑定道路');
        bind.addEventListener('click', function () {
          state.selectedRegionForRoad = region.id;
          setPicking(true, 'observation-region');
          message('vision-message', '请在地图上点击该观察区域对应的道路。');
        });
        chip.appendChild(bind);
        if (region.road_link) {
          var clearRoad = document.createElement('button'); clearRoad.type = 'button'; clearRoad.className = 'region-clear-road';
          clearRoad.textContent = '清除道路';
          clearRoad.addEventListener('click', function () {
            delete region.road_link; delete region.road_point_gcj;
            renderObservationRegionList();
          });
          chip.appendChild(clearRoad);
        }
      }
      var remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '×';
      remove.setAttribute('aria-label', '移除' + regionKindLabel(region.kind));
      remove.addEventListener('click', function () {
        state.observationRegions.splice(index, 1);
        if (!state.observationRegions.length) state.observationSceneId = null;
        renderObservationRegionList(); drawObservationRegions();
      });
      chip.appendChild(remove); root.appendChild(chip);
    });
  }

  function finishObservationRegion() {
    if (state.currentRegionPoints.length < 3) {
      message('vision-message', '一个观察区域至少需要 3 个顶点。'); return false;
    }
    state.observationRegions.push({
      id: 'region-' + Date.now() + '-' + (state.observationRegions.length + 1),
      kind: byId('observation-kind').value,
      polygon: state.currentRegionPoints.slice(),
    });
    state.currentRegionPoints = []; state.drawingRegion = false;
    renderObservationRegionList(); drawObservationRegions();
    return true;
  }

  function signatureDistance(left, right) {
    if (!/^[0-9a-f]{16}$/i.test(left || '') || !/^[0-9a-f]{16}$/i.test(right || '')) return Infinity;
    var distance = 0;
    for (var index = 0; index < 16; index += 1) {
      var value = parseInt(left[index], 16) ^ parseInt(right[index], 16);
      while (value) { distance += value & 1; value >>= 1; }
    }
    return distance;
  }

  function observationFrameSignature() {
    var image = byId('observation-image');
    var video = byId('observation-video');
    var source = video && !video.hidden && video.videoWidth ? video : image && !image.hidden && image.naturalWidth ? image : null;
    if (!source) throw new Error('请先选择可读取的图片或视频画面。');
    if (source === video && Number.isFinite(Number(video.currentTime)) && Number(video.currentTime) > 0.5) {
      throw new Error('场景画面指纹使用视频首帧，请把视频拖回开头后再保存或载入。');
    }
    var canvas = document.createElement('canvas');
    canvas.width = 9; canvas.height = 8;
    var context = canvas.getContext && canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('浏览器暂时无法生成画面指纹，请重新载入影像。');
    try { context.drawImage(source, 0, 0, 9, 8); }
    catch (_) { throw new Error('当前画面还未解码完成，请稍等后重试。'); }
    var pixels = context.getImageData(0, 0, 9, 8).data;
    var gray = [];
    for (var pixel = 0; pixel < 72; pixel += 1) {
      var offset = pixel * 4;
      gray.push(0.299 * pixels[offset] + 0.587 * pixels[offset + 1] + 0.114 * pixels[offset + 2]);
    }
    var hex = '0123456789abcdef';
    var nibble = 0;
    var nibbleBits = 0;
    var signature = '';
    for (var y = 0; y < 8; y += 1) {
      for (var x = 0; x < 8; x += 1) {
        nibble = (nibble << 1) | (gray[y * 9 + x] > gray[y * 9 + x + 1] ? 1 : 0);
        nibbleBits += 1;
        if (nibbleBits === 4) {
          signature += hex[nibble]; nibble = 0; nibbleBits = 0;
        }
      }
    }
    return signature;
  }

  async function refreshVisionScenes() {
    var select = byId('vision-scene-select');
    if (!select) return;
    try {
      var data = await request('/api/manager/vision-scenes', 'GET');
      state.observationScenes = Array.isArray(data.scenes) ? data.scenes : [];
      var selected = state.observationSceneId || select.value;
      select.replaceChildren();
      var placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = '选择观察场景';
      select.appendChild(placeholder);
      state.observationScenes.forEach(function (scene) {
        var option = document.createElement('option'); option.value = scene.scene_id;
        option.textContent = scene.name + ' · ' + (scene.observation_regions || []).length + ' 个区域';
        select.appendChild(option);
      });
      select.value = selected;
    } catch (error) {
      message('vision-message', error.message || '观察场景列表暂时无法读取。');
    }
  }

  async function saveVisionScene() {
    var name = byId('vision-scene-name').value.trim();
    if (!name) { message('vision-message', '请先填写观察场景名称。'); return; }
    if (!state.anchor) { message('vision-message', '请先在地图标注影像所在区域。'); return; }
    if (!state.observationRegions.length) { message('vision-message', '请先圈选至少一个观察区域。'); return; }
    var unbound = state.observationRegions.find(function (region) {
      return ['vehicle_lane', 'pedestrian', 'road_surface'].includes(region.kind) && !region.road_link;
    });
    if (unbound) { message('vision-message', regionKindLabel(unbound.kind) + '尚未绑定对应道路。'); return; }
    try {
      state.frameSignature = observationFrameSignature();
      var scenePayload = {
        name: name, anchor_gcj: state.anchor,
        camera_stabilized: byId('camera-stabilized').checked,
        frame_signature: state.frameSignature,
        observation_regions: state.observationRegions,
      };
      if (state.observationSceneId) scenePayload.scene_id = state.observationSceneId;
      var data = await request('/api/manager/vision-scenes', 'POST', scenePayload);
      state.observationSceneId = data.scene.scene_id;
      byId('vision-scene-name').value = '';
      await refreshVisionScenes();
      byId('vision-scene-select').value = state.observationSceneId;
      message('vision-message', '观察场景已保存。画面指纹只能筛查明显变化，复用时仍要目视核对。', true);
    } catch (error) {
      message('vision-message', error.message || '观察场景保存失败。');
    }
  }

  async function loadVisionScene() {
    var scene = state.observationScenes.find(function (item) { return item.scene_id === byId('vision-scene-select').value; });
    if (!scene) { message('vision-message', '请先选择一个已保存的观察场景。'); return; }
    try {
      var signature = observationFrameSignature();
      var distance = signatureDistance(signature, scene.frame_signature);
      if (distance > 8) {
        message('vision-message', '当前画面和该场景视角差异较大（' + distance + '/64），请重新圈选区域。'); return;
      }
      state.frameSignature = signature;
      state.observationSceneId = scene.scene_id;
      state.anchor = { lng: Number(scene.anchor_gcj.lng), lat: Number(scene.anchor_gcj.lat) };
      if (state.anchorMarker) state.map.removeLayer(state.anchorMarker);
      state.anchorMarker = L.circleMarker([state.anchor.lat, state.anchor.lng], {
        radius: 8, color: '#fff', weight: 2, fillColor: '#52786e', fillOpacity: 1,
      }).addTo(state.visionLayers);
      byId('selected-anchor').textContent = '影像参考点 · ' + state.anchor.lat.toFixed(5) + ', ' + state.anchor.lng.toFixed(5);
      byId('selected-anchor').classList.add('is-set');
      byId('camera-stabilized').checked = !!scene.camera_stabilized;
      state.observationRegions = (scene.observation_regions || []).map(function (region) {
        return JSON.parse(JSON.stringify(region));
      });
      state.currentRegionPoints = []; state.drawingRegion = false;
      renderObservationRegionList(); drawObservationRegions();
      message('vision-message', '画面初筛匹配。请核对每个区域和绑定道路；提交时服务器还会再次检查。', true);
    } catch (error) {
      message('vision-message', error.message || '当前画面无法与场景匹配。');
    }
  }

  async function deleteVisionScene() {
    var sceneId = byId('vision-scene-select').value;
    if (!sceneId) { message('vision-message', '请先选择要删除的场景。'); return; }
    if (!window.confirm('删除这条常用观察场景？已提交的分析记录不受影响。')) return;
    try {
      await request('/api/manager/vision-scenes/' + encodeURIComponent(sceneId), 'DELETE');
      if (state.observationSceneId === sceneId) state.observationSceneId = null;
      await refreshVisionScenes();
      message('vision-message', '观察场景已删除。', true);
    } catch (error) { message('vision-message', error.message || '场景删除失败。'); }
  }

  function updatePublishState() {
    var hasFieldEvidence = !state.sourceJobId || byId('field-confirmation').value.trim().length >= 8;
    var impactReady = !!byId('event-action').value && selectedEventModes().length > 0;
    byId('event-cost-wrap').hidden = byId('event-action').value !== 'slowdown';
    byId('publish-event').disabled = !state.picked || !state.picked.confirmed || state.picking ||
      !byId('event-type').value || !impactReady || !hasFieldEvidence;
  }

  function selectedEventModes() {
    var select = byId('event-modes');
    if (!select || !select.options) return ['walk', 'bike', 'drive'];
    return Array.from(select.selectedOptions || []).map(function (option) { return option.value; });
  }

  function conditionPreviewQuery(picked) {
    var query = '?lng=' + encodeURIComponent(picked.lng) +
      '&lat=' + encodeURIComponent(picked.lat) +
      '&type=' + encodeURIComponent(byId('event-type').value);
    var action = byId('event-action').value;
    if (action) {
      query += '&action=' + encodeURIComponent(action) +
        '&modes=' + encodeURIComponent(selectedEventModes().join(','));
      if (action === 'slowdown') query += '&cost_multiplier=' + encodeURIComponent(byId('event-cost').value || '1.5');
    }
    return query;
  }

  function clearPickedRoad() {
    state.pickRequestId += 1;
    setPicking(false);
    state.picked = null;
    state.impactPreviewRequestId += 1;
    if (state.preview) state.map.removeLayer(state.preview);
    state.preview = null;
    state.pickFeedbackLayers.forEach(function (layer) { state.map.removeLayer(layer); });
    state.pickFeedbackLayers = [];
    byId('selected-road').textContent = '尚未选择道路';
    byId('selected-road').classList.remove('is-set');
    byId('confirm-picked-road').hidden = true;
    byId('clear-picked-road').hidden = true;
    renderImpactPreview(null);
    updatePublishState();
  }

  function clearVisionSource() {
    state.sourceJobId = null;
    state.sourceCandidateIndex = null;
    byId('vision-source-banner').hidden = true;
    byId('field-confirmation-wrap').hidden = true;
    byId('field-confirmation').required = false;
    byId('field-confirmation').value = '';
    updatePublishState();
  }

  function prepareVisionEvent(job, candidateIndex) {
    var candidateReview = (job.candidate_reviews || {})[String(candidateIndex)];
    var legacyConfirmed = job.review_status === 'confirmed' &&
      (job.review_candidate_index === candidateIndex ||
        (job.review_candidate_index == null && (job.result.candidates || []).length === 1 && candidateIndex === 0));
    if (!(candidateReview && candidateReview.status === 'confirmed') && !legacyConfirmed) return;
    clearPickedRoad();
    state.sourceJobId = job.job_id;
    state.sourceCandidateIndex = candidateIndex;
    var candidate = (job.result.candidates || [])[candidateIndex] || {};
    var summary = candidate.evidence && candidate.evidence.summary || {};
    var regionId = typeof summary.region_id === 'string' ? summary.region_id : '';
    var region = (job.observation_regions || []).find(function (item) { return item && item.id === regionId; });
    var regionKinds = { vehicle_lane: '机动车道', pedestrian: '人行区域', road_surface: '路面', parking: '停车区', exclude: '排除区域' };
    var regionNote = region ? '观察区域：' + (region.name || region.id) + '（' + (regionKinds[region.kind] || '已标注区域') + '）。' : '';
    byId('vision-source-banner').hidden = false;
    byId('vision-source-label').textContent = '已复核影像候选 ' + (candidateIndex + 1) + '：' + (job.original_name || job.job_id) + '。' + regionNote + '请在地图上选择实际受影响道路、事件类型，并填写现场核实依据。';
    byId('field-confirmation-wrap').hidden = false;
    byId('field-confirmation').required = true;
    byId('event-type').value = '';
    byId('event-action').value = '';
    Array.from(byId('event-modes').options || []).forEach(function (option) { option.selected = false; });
    byId('event-name').value = '';
    byId('event-description').value = '';
    byId('start-time').value = '';
    byId('end-time').value = localDateTimeValue(Date.now() / 1000 + 1800);
    byId('field-confirmation').value = '';
    if (job.anchor_gcj) state.map.setView([job.anchor_gcj.lat, job.anchor_gcj.lng], 17);
    setPicking(true, 'road');
    updatePublishState();
    if (byId('road-workspace') && byId('road-workspace').scrollIntoView) {
      byId('road-workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function prepareManualVisionEvent(job) {
    clearPickedRoad();
    clearVisionSource();
    state.sourceJobId = job.job_id;
    state.sourceCandidateIndex = null;
    byId('vision-source-banner').hidden = false;
    byId('vision-source-label').textContent = '人工登记来源：' + (job.original_name || job.job_id) +
      '。模型没有生成路况候选；请先查看原片，再选择受影响道路并填写现场核实依据。';
    byId('field-confirmation-wrap').hidden = false;
    byId('field-confirmation').required = true;
    byId('event-type').value = '';
    byId('event-action').value = '';
    Array.from(byId('event-modes').options || []).forEach(function (option) { option.selected = false; });
    byId('event-name').value = '';
    byId('event-description').value = '';
    byId('start-time').value = '';
    byId('end-time').value = localDateTimeValue(Date.now() / 1000 + 1800);
    byId('field-confirmation').value = '';
    if (job.anchor_gcj) state.map.setView([job.anchor_gcj.lat, job.anchor_gcj.lng], 17);
    setPicking(true, 'road');
    updatePublishState();
    if (byId('road-workspace') && byId('road-workspace').scrollIntoView) {
      byId('road-workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  async function request(path, method, body, extraHeaders) {
    var headers = {};
    if (state.csrf && method !== 'GET') headers['X-CSRF-Token'] = state.csrf;
    Object.keys(extraHeaders || {}).forEach(function (key) { headers[key] = extraHeaders[key]; });
    var rawUploadChunk = body && typeof body.size === 'number' && typeof body.arrayBuffer === 'function';
    if (rawUploadChunk) headers['Content-Type'] = 'application/octet-stream';
    else if (body !== undefined && !(body instanceof FormData)) headers['Content-Type'] = 'application/json';
    var response = await fetch(path, {
      method: method || 'GET', headers: headers,
      body: body === undefined ? undefined : (body instanceof FormData || rawUploadChunk ? body : JSON.stringify(body)),
      credentials: 'same-origin',
    });
    var payload = await response.json().catch(function () { return {}; });
    if (!response.ok) {
      if (response.status === 401 && path !== '/api/admin/login') {
        setLoggedOut('管理登录已过期，请重新登录。');
      }
      var error = new Error(payload.message || '请求失败，请重试');
      error.status = response.status;
      throw error;
    }
    return payload.data || payload;
  }

  function setLoggedIn(data) {
    state.csrf = data.csrf_token || state.csrf;
    byId('login-panel').hidden = true;
    byId('workspace').hidden = false;
    byId('logout').hidden = false;
    if (!state.map) initMap();
    setTimeout(function () { state.map.invalidateSize(); }, 80);
    refreshEvents();
    refreshVisionStatus();
    refreshVisionScenes();
    migrateLegacyDismissedVisionJobs().then(function () { refreshVisionJobs(true); });
    if (!state.poll) state.poll = window.setInterval(function () {
      if (!document.hidden) {
        refreshVisionStatus();
        migrateLegacyDismissedVisionJobs();
        refreshVisionJobs();
      }
    }, 5000);
  }

  function setLoggedOut(notice) {
    if (state.poll) { window.clearInterval(state.poll); state.poll = null; }
    state.csrf = '';
    state.reviewDrafts = {};
    clearVisionSource();
    if (state.map) clearPickedRoad();
    byId('event-type').value = '';
    byId('event-name').value = '';
    byId('event-description').value = '';
    byId('start-time').value = '';
    byId('end-time').value = '';
    clearSelectedMedia();
    byId('workspace').hidden = true;
    byId('logout').hidden = true;
    byId('login-panel').hidden = false;
    byId('password').value = '';
    message('login-message', notice || '已退出管理台。', true);
  }

  function initMap() {
    var cfg = window.WHU_WALKER_CONFIG || {};
    var center = cfg.mapCenter || [114.363, 30.5365];
    state.map = L.map('manager-map', { zoomControl: true, preferCanvas: true })
      .setView([center[1], center[0]], cfg.mapZoom || 15);
    L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', {
      subdomains: ['1', '2', '3', '4'], maxZoom: 18,
      attribution: '© <a href="https://ditu.amap.com/" target="_blank" rel="noopener">高德地图</a>',
    }).addTo(state.map);
    state.eventLayers = L.layerGroup().addTo(state.map);
    state.visionLayers = L.layerGroup().addTo(state.map);
    state.map.on('click', handleMapClick);
  }

  function setPicking(active, purpose) {
    var nextPurpose = active ? (purpose || state.pickPurpose || 'road') : null;
    if (!active || nextPurpose !== state.pickPurpose) state.pickRequestId += 1;
    state.picking = active;
    state.pickPurpose = nextPurpose;
    byId('manager-map').classList.toggle('manager-map-picking', active);
    if (state.pickPurpose === 'road' || !active) {
      byId('pick-road').textContent = active && state.pickPurpose === 'road' ? '取消选路' : '⌖ 选取路段';
      byId('pick-state').textContent = active
        ? '请在地图道路上点选，系统会吸附到正式路网'
        : '选择路段后再填写事件';
    }
    updatePublishState();
  }

  function renderImpactPreview(preview) {
    var box = byId('impact-preview');
    box.textContent = '';
    if (!preview) { box.hidden = true; return; }
    box.hidden = false;
    var heading = document.createElement('strong');
    heading.textContent = '发布影响预览';
    box.appendChild(heading);
    var extent = document.createElement('p');
    extent.textContent = (preview.road_name || '所选道路') + ' · ' +
      (preview.affected_road_segments || 0) + ' 个路段 · 约 ' +
      (preview.affected_length_m || 0) + ' 米';
    box.appendChild(extent);
    var modes = document.createElement('p');
    var labels = { walk: '步行', bike: '骑行', drive: '驾车' };
    modes.textContent = Object.keys(preview.effects_by_mode || {}).map(function (mode) {
      var effect = preview.effects_by_mode[mode] || {};
      return labels[mode] + '：' + (effect.label || '影响未知');
    }).join('；');
    box.appendChild(modes);
    var examples = Object.keys(preview.sample_routes || {}).map(function (mode) {
      var sample = preview.sample_routes[mode] || {};
      if (sample.status !== 'available') return null;
      return labels[mode] + '示例：' + sample.before_distance_m + ' 米 → ' +
        sample.after_distance_m + ' 米（' + (sample.detour_m >= 0 ? '+' : '') + sample.detour_m + ' 米）';
    }).filter(Boolean);
    if (examples.length) {
      var routeSamples = document.createElement('p');
      routeSamples.textContent = examples.join('；');
      box.appendChild(routeSamples);
    }
    var note = document.createElement('small');
    note.textContent = [
      preview.event_scope_note,
      typeof preview.active_event_count === 'number' ? '预览使用 ' + preview.active_event_count + ' 条当前生效路况' : '',
      preview.route_sample_scope || '示例路线只反映所选路段附近的局部影响，不代表全校总影响。',
    ].filter(Boolean).join('；');
    box.appendChild(note);
  }

  async function refreshImpactPreview() {
    if (!state.picked) return;
    var picked = state.picked;
    var type = byId('event-type').value;
    var requestId = ++state.impactPreviewRequestId;
    var query = conditionPreviewQuery(picked);
    var action = byId('event-action').value;
    try {
      var data = await request('/api/road-conditions/snap' + query, 'GET');
      if (requestId !== state.impactPreviewRequestId || state.picked !== picked ||
          byId('event-type').value !== type || byId('event-action').value !== action) return;
      picked.snap = data.snap;
      picked.impactPreview = data.impact_preview;
      renderImpactPreview(data.impact_preview);
      updatePublishState();
    } catch (error) {
      if (requestId !== state.impactPreviewRequestId || state.picked !== picked ||
          byId('event-type').value !== type || byId('event-action').value !== action) return;
      renderImpactPreview({
        road_name: picked.snap && picked.snap.road_name,
        affected_length_m: picked.snap && picked.snap.chain_length_m,
        affected_road_segments: picked.snap && picked.snap.edges && picked.snap.edges.length,
        effects_by_mode: {}, sample_routes: {},
        route_sample_scope: error.message || '路线影响预览暂时不可用，请核对所选路段和事件类型。',
      });
    }
  }

  async function handleMapClick(event) {
    if (!state.picking) return;
    if (state.pickPurpose === 'vision') {
      state.anchor = { lng: event.latlng.lng, lat: event.latlng.lat };
      if (state.anchorMarker) state.map.removeLayer(state.anchorMarker);
      state.anchorMarker = L.circleMarker([state.anchor.lat, state.anchor.lng], {
        radius: 8, color: '#fff', weight: 2, fillColor: '#52786e', fillOpacity: 1,
      }).addTo(state.visionLayers);
      byId('selected-anchor').textContent = '影像参考点 · ' + state.anchor.lat.toFixed(5) + ', ' + state.anchor.lng.toFixed(5);
      byId('selected-anchor').classList.add('is-set');
      updateVisionSubmitState();
      message('vision-message', '参考点已标记。识别框仍是画面坐标，发布路况时还需重新核对道路。', true);
      setPicking(false);
      return;
    }
    if (state.pickPurpose === 'observation-region') {
      var regionId = state.selectedRegionForRoad;
      var regionRequestId = ++state.pickRequestId;
      var regionLng = event.latlng.lng;
      var regionLat = event.latlng.lat;
      message('vision-message', '正在将观察区域匹配到校园路网…');
      try {
        var regionSnap = await request('/api/road-conditions/snap?lng=' + encodeURIComponent(regionLng) +
          '&lat=' + encodeURIComponent(regionLat), 'GET');
        if (regionRequestId !== state.pickRequestId || !state.picking ||
            state.pickPurpose !== 'observation-region') return;
        if (!regionSnap.snap) throw new Error('这个位置没有匹配到校园道路，请放大后重新选。');
        var targetRegion = state.observationRegions.find(function (item) { return item.id === regionId; });
        if (!targetRegion) throw new Error('观察区域已不存在，请重新选择。');
        targetRegion.road_point_gcj = { lng: regionLng, lat: regionLat };
        targetRegion.road_link = regionSnap.snap;
        state.selectedRegionForRoad = null;
        renderObservationRegionList();
        message('vision-message', '已绑定道路：' + (regionSnap.snap.road_name || '校园道路') +
          '，吸附偏移约 ' + Math.round(regionSnap.snap.dist_m || 0) + ' 米。保存场景前请核对。', true);
      } catch (error) {
        if (regionRequestId === state.pickRequestId) message('vision-message', error.message || '道路匹配失败。');
      } finally {
        if (regionRequestId === state.pickRequestId) setPicking(false);
      }
      return;
    }
    var pickRequestId = ++state.pickRequestId;
    var lng = event.latlng.lng;
    var lat = event.latlng.lat;
    message('form-message', '正在将点位匹配到校园路网…');
    var requestedEventType = byId('event-type').value;
    var requestedAction = byId('event-action').value;
    var requestedModes = selectedEventModes();
    try {
      var snapData = await request('/api/road-conditions/snap' + conditionPreviewQuery({ lng: lng, lat: lat }), 'GET');
      if (pickRequestId !== state.pickRequestId || !state.picking || state.pickPurpose !== 'road') return;
      var snap = snapData.snap;
      if (!snap) throw new Error('这个位置没有匹配到校园道路，请放大后重选。');
      var currentEventType = byId('event-type').value;
      var previewMatchesCurrentType = currentEventType === requestedEventType &&
        byId('event-action').value === requestedAction &&
        selectedEventModes().join(',') === requestedModes.join(',');
      state.picked = { lng: lng, lat: lat, snap: snap, impactPreview: previewMatchesCurrentType ? snapData.impact_preview : null, confirmed: false };
      renderImpactPreview(previewMatchesCurrentType ? snapData.impact_preview : null);
      if (state.preview) state.map.removeLayer(state.preview);
      state.pickFeedbackLayers.forEach(function (layer) { state.map.removeLayer(layer); });
      state.pickFeedbackLayers = [];
      var geometry = (snap.geometry_gcj || []).map(function (point) { return [point[1], point[0]]; });
      if (geometry.length >= 2) {
        state.preview = L.polyline(geometry, { color: '#d48943', weight: 8, opacity: .85, lineCap: 'round' }).addTo(state.map);
      } else {
        state.preview = L.circleMarker([snap.snap_lat_gcj, snap.snap_lng_gcj], {
          radius: 8, color: '#fff', weight: 2, fillColor: '#d48943', fillOpacity: 1,
        }).addTo(state.map);
      }
      var snapLng = Number(snap.snap_lng_gcj);
      var snapLat = Number(snap.snap_lat_gcj);
      if (Number.isFinite(snapLng) && Number.isFinite(snapLat)) {
        if ((snap.dist_m || 0) > .5) {
          state.pickFeedbackLayers.push(L.polyline([[lat, lng], [snapLat, snapLng]], {
            color: '#365f50', weight: 2, opacity: .9, dashArray: '4 5', interactive: false,
          }).addTo(state.map));
        }
        state.pickFeedbackLayers.push(L.circleMarker([lat, lng], {
          radius: 5, color: '#fff', weight: 2, fillColor: '#365f50', fillOpacity: 1,
        }).addTo(state.map));
        state.pickFeedbackLayers.push(L.circleMarker([snapLat, snapLng], {
          radius: 6, color: '#fff', weight: 2, fillColor: '#d48943', fillOpacity: 1,
        }).addTo(state.map));
      }
      var road = snap.road_name || '未命名道路';
      var length = snap.chain_length_m ? Math.round(snap.chain_length_m) + ' 米路段' : '已吸附到一条道路';
      byId('selected-road').textContent = road + ' · 吸附距离 ' + Number(snap.dist_m || 0).toFixed(1) + ' 米 · ' + length;
      byId('selected-road').classList.add('is-set');
      byId('confirm-picked-road').hidden = false;
      byId('clear-picked-road').hidden = false;
      updatePublishState();
      message('form-message', '请核对地图高亮路段和吸附距离，确认后才能发布。');
      setPicking(false);
      if (!previewMatchesCurrentType && currentEventType) refreshImpactPreview();
    } catch (error) {
      if (pickRequestId !== state.pickRequestId) return;
      message('form-message', error.message || '路段匹配失败，请重试。');
    }
  }

  function eventStatus(item) {
    return item.status || 'active';
  }

  function renderEventRow(item) {
    var row = document.createElement('article');
    row.className = 'event-row';
    var status = document.createElement('span');
    status.className = 'event-status ' + eventStatus(item);
    var dot = document.createElement('i'); status.appendChild(dot);
    status.appendChild(document.createTextNode({ active: '生效中', scheduled: '待生效', expired: '已结束', revoked: '已撤销' }[eventStatus(item)] || '已结束'));
    var name = document.createElement('div');
    var title = document.createElement('div'); title.className = 'event-name'; title.textContent = item.name || '未命名事件';
    var description = document.createElement('div'); description.className = 'event-sub'; description.textContent = item.description || '无补充说明';
    name.append(title, description);
    if (item.source && item.source.kind === 'vision_job') {
      var source = document.createElement('div'); source.className = 'event-sub';
      var regionText = item.source.region_id ? ' · 观察区域 ' + item.source.region_id : '';
      var associatedRoad = item.source.associated_edge && item.source.associated_edge.road_name;
      var roadLinkText = associatedRoad ? ' → ' + associatedRoad : '';
      source.textContent = '影像来源 ' + String(item.source.job_id || '').slice(0, 8) + regionText + roadLinkText + ' · ' + (item.source.field_confirmation || '未记录核实依据');
      name.appendChild(source);
    }
    if (item.audit && item.audit.length) {
      var last = item.audit[item.audit.length - 1];
      var audit = document.createElement('div'); audit.className = 'event-sub';
      audit.textContent = '最近操作：' + ({ created: '发布', updated: '修改', ended: '结束', revoked: '撤销' }[last.action] || last.action) +
        ' · ' + (last.actor || '未知') + ' · ' + (last.at ? new Date(Number(last.at) * 1000).toLocaleString('zh-CN', { hour12: false }) : '时间未知');
      name.appendChild(audit);
    }
    var type = document.createElement('div'); type.className = 'event-type'; type.textContent = item.type_label || item.type || '管制';
    var road = document.createElement('div'); road.className = 'event-meta';
    var edge = item.edge || {};
    road.textContent = (edge.road_name || '未命名道路') + (edge.chain_length_m ? ' · ' + Math.round(edge.chain_length_m) + ' 米' : '');
    var times = document.createElement('div'); times.className = 'event-meta';
    var end = item.end_time ? new Date(Number(item.end_time) * 1000).toLocaleString('zh-CN', { hour12: false }) : '持续有效';
    times.textContent = eventStatus(item) === 'revoked' ? '撤销 ' + new Date(Number(item.revoked_at || item.updated_at) * 1000).toLocaleString('zh-CN', { hour12: false }) :
      (eventStatus(item) === 'scheduled' ? '计划开始 ' + new Date(Number(item.start_time) * 1000).toLocaleString('zh-CN', { hour12: false }) : '结束 ' + end);
    var actions = document.createElement('div'); actions.className = 'row-actions';
    if (eventStatus(item) === 'active') {
      var endButton = document.createElement('button'); endButton.type = 'button'; endButton.textContent = '结束事件';
      endButton.addEventListener('click', function () { endEvent(item.id); }); actions.appendChild(endButton);
    }
    var deleteButton = document.createElement('button'); deleteButton.type = 'button'; deleteButton.textContent = '永久删除';
    deleteButton.addEventListener('click', function () { deleteEvent(item.id); }); actions.appendChild(deleteButton);
    row.append(status, name, type, road, times, actions);
    return row;
  }

  async function refreshEvents() {
    var list = byId('event-list');
    list.replaceChildren();
    var loading = document.createElement('div'); loading.className = 'empty-state'; loading.textContent = '正在读取事件记录…'; list.appendChild(loading);
    try {
      var result = await request('/api/road-conditions?all=1', 'GET');
      var items = result.conditions || [];
      byId('active-count').textContent = items.filter(function (item) { return item.status === 'active'; }).length;
      state.eventLayers.clearLayers();
      items.forEach(function (item) {
        if (item.status === 'revoked' || item.status === 'expired') return;
        var edge = item.edge || {};
        var geometry = (edge.geometry_gcj || []).map(function (point) { return [point[1], point[0]]; });
        if (geometry.length >= 2) {
          L.polyline(geometry, { color: item.status === 'active' ? '#d48943' : '#9da79c', weight: 5, opacity: .76 }).addTo(state.eventLayers);
        } else if (edge.snap && edge.snap.lat && edge.snap.lng) {
          L.circleMarker([edge.snap.lat, edge.snap.lng], { radius: 6, color: '#fff', weight: 2, fillColor: '#d48943', fillOpacity: .95 }).addTo(state.eventLayers);
        }
      });
      list.replaceChildren();
      if (!items.length) {
        var empty = document.createElement('div'); empty.className = 'empty-state'; empty.textContent = '目前没有路况记录。可在上方地图选路后发布第一条事件。'; list.appendChild(empty);
      } else {
        items.forEach(function (item) { list.appendChild(renderEventRow(item)); });
      }
    } catch (error) {
      list.replaceChildren();
      var failed = document.createElement('div'); failed.className = 'empty-state'; failed.textContent = error.message || '事件记录暂时无法读取。'; list.appendChild(failed);
    }
  }

  function readPendingVisionUpload() {
    try { return JSON.parse(window.sessionStorage.getItem(pendingVisionUploadKey) || 'null'); }
    catch (_) { return null; }
  }

  function savePendingVisionUpload(value) {
    try {
      if (value) window.sessionStorage.setItem(pendingVisionUploadKey, JSON.stringify(value));
      else window.sessionStorage.removeItem(pendingVisionUploadKey);
    } catch (_) {}
  }

  function fileMatchesPending(file, pending) {
    return !!(file && pending && file.name === pending.filename && file.size === pending.size &&
      Number(file.lastModified || 0) === Number(pending.last_modified || 0));
  }

  function restorePendingVisionSettings(file) {
    var pending = readPendingVisionUpload();
    if (!fileMatchesPending(file, pending) || !isVideoFile(file)) return;
    byId('camera-stabilized').checked = pending.camera_stabilized === true;
    byId('captured-at').value = localDateTimeValue(Number(pending.captured_at));
    state.observationRegions = Array.isArray(pending.observation_regions) ? pending.observation_regions : [];
    state.observationSceneId = pending.scene_id || null;
    renderObservationRegionList();
    message('vision-message', '已找到这段视频的未完成上传；提交后会从已上传位置继续。', true);
    if (pending.anchor && Number.isFinite(Number(pending.anchor.lng)) && Number.isFinite(Number(pending.anchor.lat))) {
      state.anchor = { lng: Number(pending.anchor.lng), lat: Number(pending.anchor.lat) };
      byId('selected-anchor').textContent = '已恢复影像参考点 · ' + state.anchor.lat.toFixed(5) + ', ' + state.anchor.lng.toFixed(5);
      byId('selected-anchor').classList.add('is-set');
    }
  }

  function localDateTimeValue(epochSeconds) {
    var date = new Date(epochSeconds * 1000);
    if (!Number.isFinite(date.getTime())) return '';
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  }

  function captureTimeEpoch() {
    var value = byId('captured-at').value;
    if (!value) return null;
    var timestamp = new Date(value).getTime();
    return Number.isFinite(timestamp) ? timestamp / 1000 : null;
  }

  function showVisionUploadProgress(offset, total) {
    var box = byId('vision-upload-progress');
    var bar = byId('vision-upload-bar');
    var label = byId('vision-upload-label');
    if (!box || !bar || !label) return;
    var percent = total > 0 ? Math.min(100, Math.floor(offset * 100 / total)) : 0;
    box.hidden = false;
    bar.value = percent;
    label.textContent = '已上传 ' + formatVisionBytes(offset) + ' / ' + formatVisionBytes(total) + '（' + percent + '%）';
  }

  function formatVisionBytes(bytes) {
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  async function uploadVideoInChunks(file, cameraStabilized, capturedAt, observationRegions, sceneId) {
    var pending = readPendingVisionUpload();
    var matchingPending = fileMatchesPending(file, pending) &&
      pending.camera_stabilized === cameraStabilized &&
      Math.abs(Number(pending.captured_at) - Number(capturedAt)) < 1 &&
      JSON.stringify(pending.observation_regions || []) === JSON.stringify(observationRegions || []) &&
      (pending.scene_id || null) === (sceneId || null) &&
      ((!pending.anchor && !state.anchor) || (pending.anchor && state.anchor &&
        Math.abs(Number(pending.anchor.lng) - state.anchor.lng) < 0.000001 &&
        Math.abs(Number(pending.anchor.lat) - state.anchor.lat) < 0.000001));
    var uploadId = matchingPending ? pending.upload_id : null;
    var offset = 0;
    if (uploadId) {
      try {
        var resumed = await request('/api/manager/vision-uploads/' + encodeURIComponent(uploadId), 'GET');
        if (resumed.completed && resumed.job) {
          savePendingVisionUpload(null);
          return { job: resumed.job };
        }
        if (resumed.size !== file.size) throw new Error('续传文件与当前选择不一致。');
        offset = resumed.offset;
      } catch (error) {
        if (error.status !== 404 && error.status !== 410) throw error;
        savePendingVisionUpload(null);
        uploadId = null;
      }
    }
    if (!uploadId) {
      var uploadMetadata = {
        filename: file.name, size: file.size, camera_stabilized: cameraStabilized,
        captured_at: capturedAt, observation_regions: observationRegions || [],
      };
      if (sceneId) uploadMetadata.scene_id = sceneId;
      if (state.anchor) {
        uploadMetadata.lng = state.anchor.lng;
        uploadMetadata.lat = state.anchor.lat;
      }
      var started = await request('/api/manager/vision-uploads', 'POST', uploadMetadata);
      uploadId = started.upload_id;
      offset = started.offset || 0;
      pending = {
        upload_id: uploadId, filename: file.name, size: file.size,
        last_modified: Number(file.lastModified || 0),
        anchor: state.anchor ? { lng: state.anchor.lng, lat: state.anchor.lat } : null,
        camera_stabilized: cameraStabilized,
        captured_at: capturedAt,
        observation_regions: observationRegions || [],
        scene_id: sceneId || null,
      };
      savePendingVisionUpload(pending);
      if (started.chunk_size) state.uploadChunkBytes = started.chunk_size;
    }

    showVisionUploadProgress(offset, file.size);
    while (offset < file.size) {
      var end = Math.min(file.size, offset + state.uploadChunkBytes);
      var chunk = file.slice(offset, end);
      var expectedOffset = end;
      var attempts = 0;
      while (true) {
        try {
          var uploaded = await request(
            '/api/manager/vision-uploads/' + encodeURIComponent(uploadId) + '/chunks', 'PUT', chunk,
            { 'Upload-Offset': String(offset) },
          );
          if (uploaded.offset !== expectedOffset) throw new Error('服务器确认的上传位置不一致。');
          offset = uploaded.offset;
          break;
        } catch (error) {
          attempts += 1;
          var remote;
          try { remote = await request('/api/manager/vision-uploads/' + encodeURIComponent(uploadId), 'GET'); }
          catch (_) { remote = null; }
          if (remote && remote.offset === expectedOffset) { offset = remote.offset; break; }
          if (remote && remote.offset !== offset) throw new Error('续传进度发生变化，请重新选择视频后继续。');
          if (attempts >= 3) throw error;
          await new Promise(function (resolve) { window.setTimeout(resolve, 300 * attempts); });
        }
      }
      showVisionUploadProgress(offset, file.size);
    }
    var complete = await request('/api/manager/vision-uploads/' + encodeURIComponent(uploadId) + '/complete', 'POST', {});
    savePendingVisionUpload(null);
    showVisionUploadProgress(file.size, file.size);
    return complete;
  }

  async function refreshVisionStatus() {
    if (state.visionStatusBusy) return;
    state.visionStatusBusy = true;
    try {
      var data = await request('/api/manager/vision-status', 'GET');
      state.visionCapabilities = data.capabilities || {};
      var optionalModels = data.optional_model_status || {};
      function modelLabel(status) {
        return {
          ready: '已启用', not_configured: '未配置', file_missing: '权重文件缺失',
          disabled: '尚未验收，暂不启用',
          worker_not_ready: '视觉进程未就绪', load_failed: '权重未能加载',
        }[status] || '状态未知';
      }
      var modelState = '事故模型' + modelLabel(optionalModels.accident) +
        ' · 积水模型' + modelLabel(optionalModels.flood);
      byId('vision-status').textContent = data.notice + ' ' + modelState + '。图片不超过 ' + Math.round((data.max_image_bytes || data.max_media_bytes || state.maxImageBytes) / (1024 * 1024)) + ' MB；视频不超过 ' + Math.round((data.max_video_bytes || state.maxVideoBytes) / (1024 * 1024)) + ' MB，支持分块续传。';
      state.maxImageBytes = data.max_image_bytes || data.max_media_bytes || state.maxImageBytes;
      state.maxVideoBytes = data.max_video_bytes || state.maxVideoBytes;
      state.uploadChunkBytes = data.upload_chunk_bytes || state.uploadChunkBytes;
      state.inferenceReady = !!data.inference_ready;
      updateVisionSubmitState();
      updateVisionGuidance();
      byId('vision-status').classList.toggle('is-ready', !!data.inference_ready);
    } catch (error) {
      state.inferenceReady = false;
      updateVisionSubmitState();
      updateVisionGuidance();
      byId('vision-status').classList.toggle('is-ready', false);
      byId('vision-status').textContent = error.message || '视觉任务状态暂不可用。';
    } finally { state.visionStatusBusy = false; }
  }

  function candidateName(kind, item) {
    var summary = item && item.evidence && item.evidence.summary || {};
    if (kind === 'possible_flooding' && summary.water_evidence_type === 'visible_water') {
      return '路面区域可见水体线索';
    }
    return {
      possible_congestion: '可能存在车辆排队/低速聚集',
      possible_crowding: '人行区域人群聚集候选',
      possible_accident: '疑似事故场景（位置待复核）',
      possible_flooding: '疑似道路积水',
      vehicle_cluster_review: '车辆密集观察',
    }[kind] || '影像候选';
  }

  function appendVisionPreview(card, job, result) {
    if (job.media_kind === 'video') {
      var video = document.createElement('video');
      video.controls = true; video.preload = 'metadata'; video.className = 'vision-preview'; video.src = job.media_url;
      card.appendChild(video);
      return video;
    }
    var imageInfo = result.media || {};
    var width = Number(imageInfo.width), height = Number(imageInfo.height);
    var detections = Array.isArray(result.preview_detections) ? result.preview_detections.slice(0, 250) : [];
    var stage = document.createElement('div'); stage.className = 'vision-image-stage';
    var image = document.createElement('img');
    image.alt = '管理员上传的校园巡查影像'; image.className = 'vision-preview'; image.src = job.media_url;
    stage.appendChild(image);
    if (width > 0 && height > 0 && width * height <= 12500000 && detections.length) {
      var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      svg.setAttribute('preserveAspectRatio', 'none');
      svg.setAttribute('aria-label', '车辆检测框');
      var allowedLabels = {
        car: 'car', van: 'van', truck: 'truck', bus: 'bus', motor: 'motor',
        tricycle: 'tricycle', 'awning-tricycle': 'awning-tricycle', bicycle: 'bicycle',
      };
      detections.forEach(function (item) {
        if (!item || !Array.isArray(item.box) || item.box.length !== 4) return;
        var sourceLabel = String(item.label || '').trim().toLowerCase();
        if (!Object.prototype.hasOwnProperty.call(allowedLabels, sourceLabel)) return;
        var values = item.box.map(Number);
        if (!values.every(Number.isFinite)) return;
        var x1 = Math.max(0, Math.min(width, values[0]));
        var y1 = Math.max(0, Math.min(height, values[1]));
        var x2 = Math.max(0, Math.min(width, values[2]));
        var y2 = Math.max(0, Math.min(height, values[3]));
        if (x2 <= x1 || y2 <= y1) return;
        var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x1); rect.setAttribute('y', y1);
        rect.setAttribute('width', x2 - x1); rect.setAttribute('height', y2 - y1);
        svg.appendChild(rect);
        var label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x1); label.setAttribute('y', Math.max(13, y1 - 3));
        var confidence = Number(item.confidence);
        confidence = Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : 0;
        label.textContent = allowedLabels[sourceLabel] + ' ' + Math.round(confidence * 100) + '%';
        svg.appendChild(label);
      });
      stage.appendChild(svg);
    }
    var floodCandidates = Array.isArray(result.candidates) ? result.candidates.filter(function (item) {
      return item && item.kind === 'possible_flooding';
    }) : [];
    floodCandidates.forEach(function (item) {
      var outlines = item.evidence && item.evidence.summary && item.evidence.summary.outline_polygons;
      if (!Array.isArray(outlines) || !outlines.length || width <= 0 || height <= 0) return;
      var overlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      overlay.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      overlay.setAttribute('preserveAspectRatio', 'none');
      overlay.setAttribute('aria-label', '疑似积水像素轮廓，仅供管理员复核');
      outlines.slice(0, 8).forEach(function (polygon) {
        if (!Array.isArray(polygon) || polygon.length < 3) return;
        var points = polygon.map(function (point) {
          if (!Array.isArray(point) || point.length !== 2) return null;
          var x = Number(point[0]), y = Number(point[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) return null;
          return (x * width) + ',' + (y * height);
        });
        if (points.some(function (point) { return point === null; })) return;
        var shape = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        shape.setAttribute('points', points.join(' '));
        shape.setAttribute('class', 'flood-mask-outline');
        overlay.appendChild(shape);
      });
      if (overlay.childNodes && overlay.childNodes.length) stage.appendChild(overlay);
    });
    card.appendChild(stage);
    return null;
  }

  function appendCandidateEvidence(candidate, mediaElement, mediaKind, target) {
    var evidence = candidate && candidate.evidence || {};
    var segments = Array.isArray(evidence.segments) ? evidence.segments : [];
    var clips = Array.isArray(evidence.clips) ? evidence.clips.slice(0, 3) : [];
    if (mediaKind !== 'video' || (!mediaElement && !clips.length) || (!segments.length && !clips.length)) return;
    var details = document.createElement('details'); details.className = 'candidate-evidence';
    var summary = document.createElement('summary');
    summary.textContent = clips.length
      ? '查看证据（' + segments.length + ' 个时段 · ' + clips.length + ' 段可播放片段）'
      : '查看证据时段（' + segments.length + ' 段）';
    details.appendChild(summary);
    var list = document.createElement('div'); list.className = 'candidate-evidence-list';
    segments.forEach(function (segment, index) {
      var start = Number(segment && segment.start_seconds);
      var end = Number(segment && segment.end_seconds);
      if (!Number.isFinite(start) || start < 0 || !Number.isFinite(end) || end < start) return;
      var jump = document.createElement('button'); jump.type = 'button';
      jump.className = 'outline-button candidate-evidence-jump';
      jump.textContent = '片段 ' + (index + 1) + ' · ' + start.toFixed(1) + '–' + end.toFixed(1) + ' 秒';
      jump.addEventListener('click', function () {
        mediaElement.currentTime = start;
        if (typeof mediaElement.play === 'function') {
          var playback = mediaElement.play();
          if (playback && typeof playback.catch === 'function') playback.catch(function () {});
        }
      });
      list.appendChild(jump);
    });
    clips.forEach(function (clip, index) {
      var url = String(clip && clip.media_url || '');
      if (!/^\/api\/manager\/vision-jobs\/[a-f0-9-]+\/media\?clip=\d{1,4}:\d{1,2}$/i.test(url)) return;
      var start = Number(clip.start_seconds), end = Number(clip.end_seconds);
      var video = document.createElement('video');
      video.controls = true; video.preload = 'metadata'; video.playsInline = true;
      video.className = 'candidate-evidence-clip'; video.src = url;
      video.setAttribute('aria-label', '识别证据片段 ' + (index + 1));
      list.appendChild(video);
      var caption = document.createElement('small'); caption.className = 'candidate-evidence-caption';
      caption.textContent = Number.isFinite(start) && Number.isFinite(end) && end >= start
        ? '证据片段 ' + (index + 1) + ' · 原片 ' + start.toFixed(1) + '–' + end.toFixed(1) + ' 秒'
        : '证据片段 ' + (index + 1);
      list.appendChild(caption);
    });
    if (list.children.length) {
      details.appendChild(list);
      target.appendChild(details);
    }
  }

  function appendCandidateSummary(candidate, mediaKind, target) {
    var evidence = candidate && candidate.evidence || {};
    var summary = evidence.summary || {};
    var regionId = summary.region_id || evidence.region_id;
    var line = typeof regionId === 'string' && regionId ? '观察区域：' + regionId + '。' : '';
    if (candidate.kind === 'possible_accident') {
      var accidentScore = Number(summary.peak_traffic_accident_probability);
      line += '模型在 ' + Number(summary.supporting_frames || 0) + ' 帧中给出事故场景线索' +
        (Number.isFinite(accidentScore) ? '，最高分 ' + Math.round(accidentScore * 100) + '%' : '') +
        '；此模型不能定位事故车辆。';
    } else if (candidate.kind === 'possible_flooding') {
      var visibleWaterOnly = summary.water_evidence_type === 'visible_water';
      var ratio = Number(visibleWaterOnly ? summary.max_visible_water_area_ratio : summary.max_flooded_road_area_ratio);
      line += (visibleWaterOnly ? '圈选路面画面内可见水体像素约占 ' : '圈选路面中疑似淹水像素约占 ') +
        (Number.isFinite(ratio) ? Math.round(ratio * 100) + '%' : '未计算') +
        (visibleWaterOnly
          ? '；模型无法区分路面积水与其他可见水体，须人工核对。'
          : '；这是画面比例，不是水深或实际淹水面积。') +
        (mediaKind === 'image' && summary.outline_polygons && summary.outline_polygons.length ? ' 原图已叠加疑似轮廓。' : '');
    } else if (candidate.kind === 'possible_congestion') {
      var observedSeconds = Number(summary.observed_duration_seconds);
      var meanVehicles = Number(summary.mean_vehicle_count);
      var stationaryRatio = Number(summary.stationary_track_ratio);
      line += '稳定画面连续观察 ' + (Number.isFinite(observedSeconds) ? observedSeconds.toFixed(1) : '未知') +
        ' 秒；车道平均检出 ' + (Number.isFinite(meanVehicles) ? meanVehicles.toFixed(1) : '未知') +
        ' 辆，低位移车辆轨迹占 ' + (Number.isFinite(stationaryRatio) ? Math.round(stationaryRatio * 100) + '%' : '未知') +
        '。这只是拥堵候选，仍需排除信号灯停车并核对现场。';
    } else if (candidate.kind === 'possible_crowding') {
      var peakPeople = Number(evidence.peak_person_count);
      var groupedPeople = Number(evidence.peak_people_with_nearby_peer_count);
      var observedCrowdSeconds = Number(evidence.observed_duration_seconds);
      var movement = {
        mostly_moving_in_image: '多数轨迹有画面位移',
        mostly_stationary_in_image: '多数轨迹在画面内变化较少',
        camera_motion_uncompensated: '镜头运动未校正',
        insufficient_track_data: '轨迹不足，无法判断移动情况',
      }[evidence.movement_assessment] || '移动情况未知';
      line += '稳定画面连续观察 ' + (Number.isFinite(observedCrowdSeconds) ? observedCrowdSeconds.toFixed(1) : '未知') +
        ' 秒；区域内峰值 ' + (Number.isFinite(peakPeople) ? peakPeople : '未知') +
        ' 人；画面内有邻近同伴的检测数 ' + (Number.isFinite(groupedPeople) ? groupedPeople : '未知') +
        ' 人；' + movement + '。画面邻近关系和位移不代表地面密度或实际速度。';
    }
    if (!line) return;
    var note = document.createElement('small'); note.className = 'candidate-evidence-caption';
    note.textContent = line;
    target.appendChild(note);
  }

  function appendObservationRegionDiagnostics(summary, result, mediaKind, job) {
    var metrics = result.metrics || {};
    var ratios = metrics.observation_region_mapping_valid_ratios || {};
    var regionList = Array.isArray(job.observation_regions) ? job.observation_regions : [];
    var regionIds = Object.keys(ratios);
    var validRatios = regionIds.map(function (regionId) {
      return { id: regionId, ratio: Number(ratios[regionId]) };
    }).filter(function (item) {
      return Number.isFinite(item.ratio) && item.ratio >= 0 && item.ratio <= 1;
    });
    var mappingBlocked = result.safety && result.safety.observation_region_reassociation_required === true;
    if (!validRatios.length && !mappingBlocked) return false;

    var panel = document.createElement('div'); panel.className = 'vision-region-mapping';
    var heading = document.createElement('strong');
    heading.textContent = '观察区域映射诊断'; panel.appendChild(heading);
    var note = document.createElement('small');
    note.textContent = '比例表示抽样帧中区域映射成功的占比，不是地面坐标或地图定位精度。';
    panel.appendChild(note);
    validRatios.forEach(function (item) {
      var region = regionList.find(function (candidate) { return candidate && candidate.id === item.id; });
      var label = region
        ? regionKindLabel(region.kind) + '（' + item.id + '）'
        : '观察区域（' + item.id + '）';
      var row = document.createElement('small');
      row.textContent = label + '：' + (item.ratio * 100).toFixed(1) + '% 抽样帧映射成功' +
        (item.ratio < 0.8
          ? '，低于 80% 安全门槛；该区域不会生成拥堵/人流候选。'
          : '；仍需人工核对对应道路。');
      panel.appendChild(row);
    });
    if (!validRatios.length && mappingBlocked) {
      var unavailable = document.createElement('small');
      unavailable.textContent = '至少一个观察区域未达到 80% 映射安全门槛；对应区域不会生成拥堵/人流候选。';
      panel.appendChild(unavailable);
    }
    if (mappingBlocked && mediaKind === 'video') {
      var action = document.createElement('button'); action.type = 'button';
      action.className = 'outline-button vision-remap-action';
      action.textContent = '重新圈选并重新分析';
      action.addEventListener('click', retryVisionSelection);
      panel.appendChild(action);
      var actionNote = document.createElement('small');
      actionNote.textContent = '重新选择同一视频后，在首帧重画观察区并检查道路绑定。';
      panel.appendChild(actionNote);
    }
    summary.appendChild(panel);
  }

  function appendVisionMetrics(card, result, mediaKind, job) {
    var metrics = result.metrics || {};
    var safety = result.safety || {};
    var summary = document.createElement('div'); summary.className = 'vision-metrics';
    var count = document.createElement('strong');
    count.textContent = mediaKind === 'video'
      ? '每帧平均检出车辆 ' + Number(metrics.mean_vehicle_count || 0)
      : '检出车辆 ' + Number(metrics.peak_vehicle_count || 0);
    summary.appendChild(count);
    if (mediaKind === 'video') {
      var peak = document.createElement('span'); peak.textContent = '抽样帧 ' + Number(metrics.frames_analyzed || 0) + ' · 单帧最多 ' + Number(metrics.peak_vehicle_count || 0);
      summary.appendChild(peak);
      var motion = document.createElement('span');
      motion.textContent = {
        camera_motion_compensated: '轨迹运动补偿通过，道路区域仍需核对',
        operator_declared_stabilized: '按管理员声明使用固定/已稳像视角',
        camera_motion_uncompensated: '镜头运动校正不足，仅供查看车辆数量',
        insufficient_single_frame: '画面数量不足',
      }[metrics.motion_assessment] || '镜头运动状态未知';
      summary.appendChild(motion);
      if (safety.observation_region_reassociation_required === true) {
        var regionReview = document.createElement('small');
        regionReview.textContent = '镜头移动后仍需人工核对观察区域与道路的对应关系；画面跟踪不等于地图定位。';
        summary.appendChild(regionReview);
      }
    }
    appendObservationRegionDiagnostics(summary, result, mediaKind, job || {});
    var counts = metrics.peak_class_counts || {};
    var classes = Object.keys(counts).filter(function (label) { return Number(counts[label]) > 0; }).slice(0, 8);
    if (classes.length) {
      var list = document.createElement('span');
      list.textContent = '类别峰值：' + classes.map(function (label) { return label + ' ' + counts[label]; }).join('、');
      summary.appendChild(list);
    }
    var model = result.model || {};
    var accident = model.accident_model;
    var flood = model.flood_model;
    var optionalErrors = Array.isArray(result.optional_model_errors) ? result.optional_model_errors : [];
    function optionalModelLabel(name, record, enabled) {
      if (optionalErrors.some(function (item) { return item && item.model === name; }) ||
          record && record.runtime_status === 'inference_failed') return '推理失败';
      if (!enabled) return '未启用';
      return String(record && record.version || '已加载');
    }
    var version = document.createElement('small');
    version.textContent = '人车模型 ' + String(model.id || '未记录') + ' · ' + String(model.version || '未记录') +
      '；事故场景模型 ' + optionalModelLabel('accident', accident, safety.accident_recognition_supported) +
      '；路面积水分割模型 ' + optionalModelLabel('flood', flood, safety.flood_segmentation_supported) +
      '。识别框不是地面坐标，事故分类不能定位事故车辆。';
    summary.appendChild(version);
    card.appendChild(summary);
  }

  function reviewJob(job, status, candidateIndex, noteField, feedback) {
    var note = noteField.value.trim();
    if (note.length < 8) {
      feedback.textContent = '请先填写至少 8 字的影像复核依据。';
      return;
    }
    feedback.textContent = '正在保存审核…';
    var body = { status: status, note: note };
    request('/api/manager/vision-jobs/' + encodeURIComponent(job.job_id) + '/candidates/' + candidateIndex + '/review', 'POST', body)
      .then(function () {
        state.reviewDrafts[job.job_id] = '';
        return refreshVisionJobs(true);
      })
      .catch(function (error) { feedback.textContent = error.message || '审核保存失败。'; });
  }

  async function recoverVisionJob(job, action) {
    var label = action === 'cancel' ? '取消该影像任务？' : '从最近完成的视频分段重新排队？';
    if (!window.confirm(label)) return;
    try {
      await request('/api/manager/vision-jobs/' + encodeURIComponent(job.job_id) + '/' + action, 'POST', {});
      await refreshVisionJobs(true);
    } catch (error) { window.alert(error.message || '影像任务操作失败。'); }
  }

  function renderVisionJob(job) {
    var card = document.createElement('article'); card.className = 'vision-job';
    var head = document.createElement('div'); head.className = 'vision-job-head';
    var title = document.createElement('div'); title.className = 'vision-job-title'; title.textContent = job.original_name || '巡查影像';
    var meta = document.createElement('div'); meta.className = 'vision-job-meta';
    var captured = job.captured_at ? new Date(job.captured_at).toLocaleString('zh-CN', { hour12: false }) : '拍摄时间未记录';
    meta.textContent = '拍摄 ' + captured + (job.anchor_gcj ? ' · 校园位置已标注' : '')
      + (job.media_kind === 'video' ? (job.camera_stabilized ? ' · 管理员声明固定/已稳像' : ' · 自动检查镜头运动') : '');
    title.appendChild(meta);
    var stateLabel = document.createElement('span'); stateLabel.className = 'vision-job-state ' + job.status;
    stateLabel.textContent = {
      queued: '排队中', running: job.cancel_requested ? '等待取消' : '分析中', needs_review: job.review_status ? '已复核' : '待复核',
      completed: '完成', failed: '分析失败', cancelled: '已取消',
    }[job.status] || job.status;
    head.append(title, stateLabel); card.appendChild(head);
    if (job.status === 'running' || job.status === 'queued') {
      var progressInfo = job.progress || {};
      var progressLine = document.createElement('p'); progressLine.className = 'vision-job-meta';
      var parts = [];
      if (progressInfo.percent != null && Number.isFinite(Number(progressInfo.percent))) parts.push(Number(progressInfo.percent) + '%');
      if (progressInfo.frames_analyzed != null) {
        var sampledTotal = Number(progressInfo.total_sampled_frames);
        parts.push(Number(progressInfo.frames_analyzed) + (Number.isFinite(sampledTotal) && sampledTotal > 0
          ? ' / ' + sampledTotal + ' 个抽样帧' : ' 个抽样帧'));
      }
      if (progressInfo.analyzed_through_seconds != null) parts.push('已分析到 ' + Number(progressInfo.analyzed_through_seconds).toFixed(1) + ' 秒');
      progressLine.textContent = parts.length ? parts.join(' · ') : '等待视觉工作进程';
      card.appendChild(progressLine);
      if (progressInfo.percent != null && Number.isFinite(Number(progressInfo.percent))) {
        var progressBar = document.createElement('progress'); progressBar.max = 100; progressBar.value = Number(progressInfo.percent);
        progressBar.className = 'vision-job-progress'; card.appendChild(progressBar);
      }
      var cancelButton = document.createElement('button'); cancelButton.type = 'button'; cancelButton.className = 'outline-button';
      cancelButton.textContent = job.cancel_requested ? '正在等待分段结束' : '取消分析';
      cancelButton.disabled = !!job.cancel_requested;
      cancelButton.addEventListener('click', function () { recoverVisionJob(job, 'cancel'); });
      card.appendChild(cancelButton);
    }
    if (job.status === 'failed' || job.status === 'cancelled') {
      var retryButton = document.createElement('button'); retryButton.type = 'button'; retryButton.className = 'outline-button';
      retryButton.textContent = '从断点重试'; retryButton.addEventListener('click', function () { recoverVisionJob(job, 'retry'); });
      card.appendChild(retryButton);
    }
    if (job.status === 'failed' && job.error) {
      var failure = document.createElement('p'); failure.className = 'vision-job-meta';
      failure.textContent = job.error.message || '分析失败'; card.appendChild(failure);
    }
    if (job.review_status) {
      var reviewed = document.createElement('p'); reviewed.className = 'vision-job-meta';
      reviewed.textContent = (job.review_status === 'confirmed' ? '已确认' : '已排除') +
        ' · ' + (job.reviewed_by || '管理者') +
        (job.reviewed_at ? ' · ' + new Date(job.reviewed_at).toLocaleString('zh-CN', { hour12: false }) : '') +
        (job.review_note ? ' · ' + job.review_note : '');
      card.appendChild(reviewed);
    }
    var result = job.result || {};
    if (job.result && (job.status === 'needs_review' || job.status === 'completed' || job.review_status)) {
      var previewMedia = appendVisionPreview(card, job, result);
      appendVisionMetrics(card, result, job.media_kind, job);
      var reviewNote = null;
      var reviewFeedback = null;
      if (job.status === 'needs_review' && !job.review_status && (result.candidates || []).length) {
        var noteLabel = document.createElement('label'); noteLabel.className = 'vision-review-label';
        noteLabel.textContent = '影像复核依据';
        reviewNote = document.createElement('textarea'); reviewNote.maxLength = 1000;
        reviewNote.className = 'vision-review-note';
        reviewNote.value = state.reviewDrafts[job.job_id] || '';
        reviewNote.addEventListener('input', function () {
          state.reviewDrafts[job.job_id] = reviewNote.value;
        });
        reviewNote.placeholder = '说明查看了哪些影像证据，以及确认或排除的理由（至少 8 字）';
        noteLabel.appendChild(reviewNote); card.appendChild(noteLabel);
        reviewFeedback = document.createElement('p'); reviewFeedback.className = 'form-message';
        card.appendChild(reviewFeedback);
      }
      var candidates = document.createElement('div'); candidates.className = 'candidate-list';
      (result.candidates || []).forEach(function (item, index) {
        var candidateReview = (job.candidate_reviews || {})[String(index)] || null;
        var legacyReview = job.review_status && (job.review_candidate_index === index || ((result.candidates || []).length === 1 && job.review_candidate_index == null));
        var candidate = document.createElement('div'); candidate.className = 'candidate-card';
        var titleNode = document.createElement('strong');
        titleNode.textContent = candidateName(item.kind, item) + (item.kind === 'possible_flooding'
          ? ' · 需人工核对'
          : ' · 置信度 ' + Math.round((item.confidence || 0) * 100) + '%');
        if (candidateReview) titleNode.textContent += candidateReview.status === 'confirmed' ? ' · 已确认' : ' · 已排除';
        else if (legacyReview) titleNode.textContent += job.review_status === 'confirmed' ? ' · 已确认' : ' · 已排除';
        var why = document.createElement('p'); why.textContent = item.reason || '请结合原始影像复核。';
        candidate.append(titleNode, why);
        appendCandidateSummary(item, job.media_kind, candidate);
        appendCandidateEvidence(item, previewMedia, job.media_kind, candidate);
        if (candidateReview && candidateReview.note) {
          var reviewText = document.createElement('p'); reviewText.textContent = '复核记录：' + candidateReview.note; candidate.appendChild(reviewText);
        }
        if (job.status === 'needs_review' && !job.review_status && !candidateReview && !legacyReview) {
          var actions = document.createElement('div'); actions.className = 'candidate-actions';
          var confirmButton = document.createElement('button'); confirmButton.type = 'button'; confirmButton.textContent = '确认此候选';
          confirmButton.addEventListener('click', function () {
            reviewJob(job, 'confirmed', index, reviewNote, reviewFeedback);
          });
          var dismissCandidate = document.createElement('button'); dismissCandidate.type = 'button'; dismissCandidate.textContent = '排除此候选';
          dismissCandidate.addEventListener('click', function () {
            reviewJob(job, 'dismissed', index, reviewNote, reviewFeedback);
          });
          actions.append(confirmButton, dismissCandidate); candidate.appendChild(actions);
        }
        if ((candidateReview && candidateReview.status === 'confirmed') ||
            (legacyReview && job.review_status === 'confirmed')) {
          var transfer = document.createElement('button'); transfer.type = 'button';
          transfer.className = 'outline-button transfer-button'; transfer.textContent = '将此候选转入道路事件';
          transfer.addEventListener('click', function () { prepareVisionEvent(job, index); });
          candidate.appendChild(transfer);
        }
        candidates.appendChild(candidate);
      });
      if (candidates.children.length) card.appendChild(candidates);
      if (reviewNote) {
        var dismissButton = document.createElement('button'); dismissButton.type = 'button';
        dismissButton.className = 'outline-button dismiss-vision';
        dismissButton.textContent = '排除整条影像任务';
        dismissButton.addEventListener('click', function () {
          reviewJob(job, 'dismissed', null, reviewNote, reviewFeedback);
        });
        card.appendChild(dismissButton);
      }
      if (job.review_status === 'confirmed') {
        var next = document.createElement('p'); next.className = 'vision-job-meta';
        next.textContent = '已确认影像迹象。若需影响导航，请在上方路况表单中手工选定具体道路并发布事件。';
        card.appendChild(next);
        if (Number.isInteger(job.review_candidate_index) || (result.candidates || []).length === 1) {
          var transfer = document.createElement('button'); transfer.type = 'button';
          transfer.className = 'outline-button transfer-button'; transfer.textContent = '转入道路事件';
          transfer.addEventListener('click', function () { prepareVisionEvent(job); });
          card.appendChild(transfer);
        } else {
          var ambiguous = document.createElement('p'); ambiguous.className = 'vision-job-meta';
          ambiguous.textContent = '旧审核未记录具体候选，无法直接作为道路事件来源。';
          card.appendChild(ambiguous);
        }
      }
      if (job.status === 'completed' && !(result.candidates || []).length) {
        var terminal = document.createElement('div'); terminal.className = 'vision-terminal';
        var terminalTitle = document.createElement('strong'); terminalTitle.textContent = '分析已结束';
        var terminalCopy = document.createElement('p');
        terminalCopy.textContent = result.safety && result.safety.observation_region_reassociation_required === true
          ? '观察区域映射成功率低于 80%，系统不会基于相关区域生成拥堵或人群候选。请重新圈选观察区、核对道路绑定后重新分析；画面映射不等于地图地理配准。'
          : '模型未生成可复核的路况候选。若查看原片后发现遗漏，可由管理员手工登记；车辆数量不能单独证明拥堵。';
        terminal.append(terminalTitle, terminalCopy); card.appendChild(terminal);
        var manual = document.createElement('button'); manual.type = 'button';
        manual.className = 'outline-button transfer-button';
        manual.textContent = '依据原片手工登记路况';
        manual.addEventListener('click', function () { prepareManualVisionEvent(job); });
        card.appendChild(manual);
        if (!(result.safety && result.safety.observation_region_reassociation_required === true)) {
          appendVisionRecoveryActions(card, job);
        }
      }
    }
    if (job.status === 'failed') appendVisionRecoveryActions(card, job);
    appendVisionDeleteAction(card, job);
    return card;
  }

  function appendVisionRecoveryActions(card, job) {
    var actions = document.createElement('div'); actions.className = 'vision-recovery-actions';
    var retry = document.createElement('button'); retry.type = 'button'; retry.textContent = '重新选择影像';
    retry.addEventListener('click', retryVisionSelection);
    actions.appendChild(retry); card.appendChild(actions);
  }

  function appendVisionDeleteAction(card, job) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'outline-button delete-vision';
    button.textContent = '删除影像记录';
    button.disabled = state.visionDeleteBusy.has(job.job_id);
    button.addEventListener('click', function () { deleteVisionJob(job.job_id); });
    card.appendChild(button);
  }

  async function refreshVisionJobs(force) {
    if (!byId('workspace') || byId('workspace').hidden) return;
    if (!force && document.activeElement && document.activeElement.classList &&
        document.activeElement.classList.contains('vision-review-note')) return;
    var root = byId('vision-jobs');
    try {
      var data = await request('/api/manager/vision-jobs?limit=20', 'GET');
      if (!force && document.activeElement && document.activeElement.classList &&
          document.activeElement.classList.contains('vision-review-note')) return;
      var jobs = (data.jobs || []).filter(function (job) { return !isVisionJobHidden(job.job_id); });
      root.replaceChildren();
      if (!jobs.length) {
        var empty = document.createElement('div'); empty.className = 'empty-state';
        empty.textContent = state.legacyDismissedVisionJobs.size
          ? '正在清理旧版标记为已清除的影像记录。'
          : '尚无巡查任务。提交影像后，分析进度和候选都会显示在这里。';
        root.appendChild(empty);
      } else { jobs.forEach(function (job) { root.appendChild(renderVisionJob(job)); }); }
    } catch (error) {
      root.replaceChildren(); var failed = document.createElement('div'); failed.className = 'empty-state';
      failed.textContent = error.message || '视觉任务记录暂时无法读取。'; root.appendChild(failed);
    }
  }

  async function submitVision(event) {
    event.preventDefault();
    if (state.visionUploadBusy) return;
    if (!state.inferenceReady) { message('vision-message', '视觉分析当前尚未就绪，请稍后重试。'); return; }
    var file = byId('media-file').files[0];
    if (!file) { message('vision-message', '请先选择一张图片或一段视频。'); return; }
    var video = isVideoFile(file);
    var capturedAt = captureTimeEpoch();
    if (capturedAt === null) {
      message('vision-message', '请填写影像实际拍摄时间。'); return;
    }
    var maxBytes = video ? state.maxVideoBytes : state.maxImageBytes;
    if (file.size > maxBytes) {
      message('vision-message', '文件超过当前上限：' + (video ? '视频 ' : '图片 ') + Math.round(maxBytes / (1024 * 1024)) + ' MB。'); return;
    }
    if (!video && !isImageFile(file)) {
      message('vision-message', '只支持 JPG、PNG、WebP 图片和 MP4、MOV、AVI、WebM 视频。'); return;
    }
    var cameraStabilized = video && byId('camera-stabilized').checked;
    var observationRegions = state.observationRegions.slice();
    var button = byId('submit-vision');
    state.visionUploadBusy = true;
    updateVisionSubmitState();
    button.textContent = video ? '正在上传视频…' : '正在提交…';
    try {
      var data;
      if (video) {
        data = await uploadVideoInChunks(file, cameraStabilized, capturedAt, observationRegions, state.observationSceneId);
      } else {
        var form = new FormData(); form.append('media', file);
        form.append('camera_stabilized', String(cameraStabilized));
        form.append('captured_at', new Date(capturedAt * 1000).toISOString());
        form.append('observation_regions', JSON.stringify(observationRegions));
        if (state.observationSceneId) form.append('scene_id', state.observationSceneId);
        if (state.anchor) {
          form.append('lng', String(state.anchor.lng));
          form.append('lat', String(state.anchor.lat));
        }
        data = await request('/api/manager/vision-jobs', 'POST', form);
      }
      message('vision-message', '影像任务已入队。分析服务独立运行，不会阻塞路线规划。', true);
      clearSelectedMedia(); refreshVisionJobs();
      state.observationRegions = []; state.currentRegionPoints = []; state.drawingRegion = false;
      state.observationSceneId = null;
      renderObservationRegionList();
      byId('observation-editor').hidden = true;
      var progress = byId('vision-upload-progress');
      if (progress) progress.hidden = true;
    } catch (error) {
      var resumable = video && readPendingVisionUpload();
      message('vision-message', resumable
        ? (error.message || '视频上传中断。') + ' 已保存上传进度，重新选择同一视频即可继续。'
        : error.message || '任务提交失败。');
    }
    finally {
      state.visionUploadBusy = false;
      button.innerHTML = '提交分析任务 <span>→</span>';
      updateVisionSubmitState();
    }
  }

  async function endEvent(id) {
    if (!window.confirm('结束这条管制事件？记录会保留在历史中。')) return;
    try { await request('/api/road-conditions/' + encodeURIComponent(id), 'PATCH', { action: 'end' }); refreshEvents(); }
    catch (error) { window.alert(error.message); }
  }

  async function deleteEvent(id) {
    if (!window.confirm('永久删除这条路段管制记录？删除后无法恢复。')) return;
    try { await request('/api/road-conditions/' + encodeURIComponent(id), 'DELETE'); refreshEvents(); }
    catch (error) { window.alert(error.message); }
  }

  async function submitEvent(event) {
    event.preventDefault();
    if (!state.picked) { message('form-message', '请先在地图上选取一条道路。'); return; }
    if (!state.picked.confirmed) { message('form-message', '请先核对并确认地图高亮的路段。'); return; }
    if (!byId('event-type').value) { message('form-message', '请选择现场确认的事件类型。'); return; }
    if (!byId('event-action').value || !selectedEventModes().length) {
      message('form-message', '请选择路况影响方式及受影响的出行方式。'); return;
    }
    var fieldConfirmation = byId('field-confirmation').value.trim();
    if (state.sourceJobId && fieldConfirmation.length < 8) {
      message('form-message', '请填写具体道路的现场核实依据（至少 8 字）。'); return;
    }
    var start = byId('start-time').value;
    var end = byId('end-time').value;
    if (start && end && end <= start) { message('form-message', '结束时间必须晚于开始时间。'); return; }
    var button = byId('publish-event'); button.disabled = true; button.textContent = '正在发布…';
    var body = {
      type: byId('event-type').value,
      action: byId('event-action').value,
      affected_modes: selectedEventModes(),
      name: byId('event-name').value.trim(),
      description: byId('event-description').value.trim(),
      lng: state.picked.lng, lat: state.picked.lat,
    };
    if (body.action === 'slowdown') body.cost_multiplier = Number(byId('event-cost').value || 1.5);
    if (start) body.start_time = new Date(start).toISOString();
    if (end) body.end_time = new Date(end).toISOString();
    if (state.sourceJobId) {
      body.source_vision_job_id = state.sourceJobId;
      body.source_vision_candidate_index = state.sourceCandidateIndex;
      body.field_confirmation = fieldConfirmation;
    }
    try {
      await request('/api/road-conditions', 'POST', body);
      message('form-message', '事件已发布，路线规划会按该路段状态处理。', true);
      byId('event-name').value = ''; byId('event-description').value = '';
      byId('start-time').value = ''; byId('end-time').value = '';
      Array.from(byId('event-modes').options || []).forEach(function (option) { option.selected = false; });
      byId('event-action').value = '';
      clearPickedRoad(); clearVisionSource();
      refreshEvents();
    } catch (error) { message('form-message', error.message || '事件发布失败。'); }
    finally { button.innerHTML = '发布事件 <span>→</span>'; updatePublishState(); }
  }

  byId('login-form').addEventListener('submit', async function (event) {
    event.preventDefault();
    message('login-message', '正在验证…');
    try {
      var data = await request('/api/admin/login', 'POST', { password: byId('password').value });
      state.csrf = data.csrf_token || '';
      setLoggedIn(data);
    } catch (error) { message('login-message', error.message || '无法登录管理台。'); }
  });
  byId('pick-road').addEventListener('click', function () { setPicking(!(state.picking && state.pickPurpose === 'road'), 'road'); });
  byId('confirm-picked-road').addEventListener('click', function () {
    if (!state.picked) return;
    state.picked.confirmed = true;
    byId('confirm-picked-road').hidden = true;
    message('form-message', '已确认高亮路段，可以填写并发布事件。', true);
    updatePublishState();
  });
  byId('clear-picked-road').addEventListener('click', function () {
    clearPickedRoad();
    message('form-message', '已清除所选路段。');
  });
  byId('event-type').addEventListener('change', function () { updatePublishState(); return refreshImpactPreview(); });
  byId('event-action').addEventListener('change', function () { updatePublishState(); return refreshImpactPreview(); });
  byId('event-modes').addEventListener('change', function () { updatePublishState(); return refreshImpactPreview(); });
  byId('event-cost').addEventListener('input', function () { updatePublishState(); return refreshImpactPreview(); });
  byId('field-confirmation').addEventListener('input', updatePublishState);
  byId('clear-vision-source').addEventListener('click', clearVisionSource);
  byId('pick-anchor').addEventListener('click', function () { setPicking(true, 'vision'); });
  byId('start-region').addEventListener('click', function () {
    if (!byId('observation-canvas').width) { message('vision-message', '请先选择可读取的图片或视频画面。'); return; }
    state.currentRegionPoints = []; state.drawingRegion = true;
    message('vision-message', '已开始圈选，至少点 3 个顶点后选择“完成区域”。', true);
    drawObservationRegions();
  });
  byId('undo-region-point').addEventListener('click', function () {
    state.currentRegionPoints.pop(); drawObservationRegions();
  });
  byId('finish-region').addEventListener('click', finishObservationRegion);
  byId('save-vision-scene').addEventListener('click', saveVisionScene);
  byId('refresh-vision-scenes').addEventListener('click', refreshVisionScenes);
  byId('load-vision-scene').addEventListener('click', loadVisionScene);
  byId('delete-vision-scene').addEventListener('click', deleteVisionScene);
  byId('clear-regions').addEventListener('click', function () {
    state.observationRegions = []; state.currentRegionPoints = []; state.drawingRegion = false;
    state.observationSceneId = null;
    renderObservationRegionList(); drawObservationRegions();
  });
  byId('observation-canvas').addEventListener('click', function (event) {
    if (!state.drawingRegion) return;
    var bounds = this.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    state.currentRegionPoints.push([
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    ]);
    drawObservationRegions();
  });
  byId('media-file').addEventListener('change', function () {
    var file = byId('media-file').files[0];
    renderSelectedMedia(file);
    state.observationSceneId = null; state.frameSignature = null;
    state.observationRegions = []; state.currentRegionPoints = []; state.drawingRegion = false;
    var pending = readPendingVisionUpload();
    if (fileMatchesPending(file, pending) && pending.captured_at) {
      byId('captured-at').value = localDateTimeValue(Number(pending.captured_at));
    } else {
      byId('captured-at').value = '';
      if (!file) {
        state.anchor = null;
        byId('selected-anchor').textContent = '尚未标注影像区域';
        byId('selected-anchor').classList.remove('is-set');
      }
    }
    renderObservationRegionList();
    showObservationMedia(file);
    byId('camera-stabilized').disabled = !isVideoFile(file);
    if (!isVideoFile(file)) byId('camera-stabilized').checked = false;
    restorePendingVisionSettings(file);
    updateVisionSubmitState();
    updateVisionGuidance();
  });
  byId('vision-form').addEventListener('submit', submitVision);
  byId('condition-form').addEventListener('submit', submitEvent);
  byId('refresh-events').addEventListener('click', refreshEvents);
  byId('logout').addEventListener('click', async function () {
    try { await request('/api/admin/logout', 'POST', {}); } catch (_) {}
    setLoggedOut('已退出管理台。');
  });

  request('/api/admin/status', 'GET').then(function (data) {
    if (data.is_admin) setLoggedIn(data);
    else if (!data.login_enabled) message('login-message', '管理功能尚未配置，请联系系统管理员。');
  }).catch(function () { message('login-message', '管理服务暂时不可用，请稍后刷新。'); });
}());
