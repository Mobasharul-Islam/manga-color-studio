document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Input Section
  const selectFolderBtn = document.getElementById('selectFolderBtn');
  const selectFilesBtn = document.getElementById('selectFilesBtn');
  const fileInput = document.getElementById('fileInput');
  const folderInput = document.getElementById('folderInput');
  const dropzone = document.getElementById('dropzone');
  const thumbnailContainer = document.getElementById('thumbnailContainer');
  const thumbnailList = document.getElementById('thumbnailList');
  const pageCountBadge = document.getElementById('pageCountBadge');
  const loadedFolderLabel = document.getElementById('loadedFolderLabel');
  const clearPagesBtn = document.getElementById('clearPagesBtn');

  // DOM Elements - LoRA Management
  let installedLoras = [];
  const loraStack = document.getElementById('loraStack');
  const toggleAddLoraBtn = document.getElementById('toggleAddLoraBtn');
  const addLoraForm = document.getElementById('addLoraForm');
  const closeAddLoraBtn = document.getElementById('closeAddLoraBtn');
  const cancelAddLoraBtn = document.getElementById('cancelAddLoraBtn');
  const submitAddLoraBtn = document.getElementById('submitAddLoraBtn');
  const newLoraUrl = document.getElementById('newLoraUrl');
  const newLoraName = document.getElementById('newLoraName');
  const newLoraScale = document.getElementById('newLoraScale');
  const newLoraIsTurbo = document.getElementById('newLoraIsTurbo');
  const newLoraCivitaiToken = document.getElementById('newLoraCivitaiToken');
  const addLoraErrorMsg = document.getElementById('addLoraErrorMsg');

  // Tabs for Add LoRA Modal
  const tabLoraLink = document.getElementById('tabLoraLink');
  const tabLoraUpload = document.getElementById('tabLoraUpload');
  const paneLoraLink = document.getElementById('paneLoraLink');
  const paneLoraUpload = document.getElementById('paneLoraUpload');
  const loraFileUploadBox = document.getElementById('loraFileUploadBox');
  const loraFileInput = document.getElementById('loraFileInput');
  const loraFileNameLabel = document.getElementById('loraFileNameLabel');
  const uploadLoraName = document.getElementById('uploadLoraName');
  const uploadLoraScale = document.getElementById('uploadLoraScale');
  const uploadLoraIsTurbo = document.getElementById('uploadLoraIsTurbo');
  const cancelUploadLoraBtn = document.getElementById('cancelUploadLoraBtn');
  const submitUploadLoraBtn = document.getElementById('submitUploadLoraBtn');
  let selectedLoraFile = null;

  // Cloud Dataset Elements
  const lorasDatasetPill = document.getElementById('lorasDatasetPill');
  const modelsDatasetPill = document.getElementById('modelsDatasetPill');
  const syncLorasBtn = document.getElementById('syncLorasBtn');
  const syncModelsBtn = document.getElementById('syncModelsBtn');

  // Prompt Section Elements & Presets
  const positivePromptInput = document.getElementById('positivePromptInput');
  const negativePromptInput = document.getElementById('negativePromptInput');
  const resetPromptBtn = document.getElementById('resetPromptBtn');
  const guidanceScaleSlider = document.getElementById('guidanceScaleSlider');
  const guidanceScaleVal = document.getElementById('guidanceScaleVal');
  const presetChips = document.querySelectorAll('.preset-chip');

  // Launch & Monitor Elements
  const startColorizeBtn = document.getElementById('startColorizeBtn');
  const openOutputBtn = document.getElementById('openOutputBtn');
  const jobPulse = document.getElementById('jobPulse');
  const jobStatusTitle = document.getElementById('jobStatusTitle');
  const jobTimer = document.getElementById('jobTimer');
  const progressBarFill = document.getElementById('progressBarFill');
  const phaseMessage = document.getElementById('phaseMessage');
  const progressPercent = document.getElementById('progressPercent');
  const terminalLog = document.getElementById('terminalLog');

  // Gallery & Split Compare Elements
  const galleryChapterSelect = document.getElementById('galleryChapterSelect');
  const tabSplitView = document.getElementById('tabSplitView');
  const tabSideBySide = document.getElementById('tabSideBySide');
  const tabGridView = document.getElementById('tabGridView');
  const splitViewContainer = document.getElementById('splitViewContainer');
  const sideBySideContainer = document.getElementById('sideBySideContainer');
  const gridViewContainer = document.getElementById('gridViewContainer');
  const comparisonSlider = document.getElementById('comparisonSlider');
  const compResizeWrap = document.getElementById('compResizeWrap');
  const sliderHandle = document.getElementById('sliderHandle');
  const compColorImg = document.getElementById('compColorImg');
  const compInputImg = document.getElementById('compInputImg');
  const sbsInputImg = document.getElementById('sbsInputImg');
  const sbsColorImg = document.getElementById('sbsColorImg');
  const pageCarousel = document.getElementById('pageCarousel');
  const sbsPageCarousel = document.getElementById('sbsPageCarousel');
  const resultsGrid = document.getElementById('resultsGrid');

  let currentPages = [];
  let resultPairs = [];
  let isPolling = false;
  let activePairIndex = 0;

  // ==========================================
  // 1. CHAPTER & IMAGE SELECTION
  // ==========================================

  function renderThumbnails(images, label) {
    if (!thumbnailContainer || !pageCountBadge || !thumbnailList) return;
    if (!images || images.length === 0) {
      thumbnailContainer.style.display = 'none';
      pageCountBadge.textContent = '0 Pages';
      return;
    }
    thumbnailContainer.style.display = 'block';
    if (loadedFolderLabel) loadedFolderLabel.textContent = `${label} (${images.length} pages)`;
    pageCountBadge.textContent = `${images.length} Pages`;
    thumbnailList.innerHTML = images.map((img, i) => `
      <div class="thumbnail-item" title="${escapeHtml(img.name)}">
        <img src="${img.url}" alt="${escapeHtml(img.name)}" loading="lazy">
      </div>
    `).join('');
  }

  // Choose Folder Button (triggers native browser directory picker or fallback)
  // High-speed pure JS ZIP generator (Store mode - 0% compression for 100% instant packaging)
  async function packFilesToZip(fileList) {
    let crcTable = packFilesToZip._crcTable;
    if (!crcTable) {
      crcTable = new Uint32Array(256);
      for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        crcTable[i] = c;
      }
      packFilesToZip._crcTable = crcTable;
    }

    function getCrc32(bytes) {
      let crc = 0 ^ (-1);
      for (let i = 0; i < bytes.length; i++) {
        crc = (crc >>> 8) ^ crcTable[(crc ^ bytes[i]) & 0xFF];
      }
      return (crc ^ (-1)) >>> 0;
    }

    const fileEntries = [];
    let offset = 0;
    const parts = [];
    const encoder = new TextEncoder();

    for (const file of fileList) {
      const name = file.name || 'image.png';
      const nameBytes = encoder.encode(name);
      const dataBuffer = await file.arrayBuffer();
      const dataBytes = new Uint8Array(dataBuffer);
      const crc = getCrc32(dataBytes);
      const size = dataBytes.length;

      // Local file header (30 bytes)
      const localHeader = new Uint8Array(30 + nameBytes.length);
      const view = new DataView(localHeader.buffer);
      view.setUint32(0, 0x04034b50, true); // Local header signature
      view.setUint16(4, 20, true);         // Version needed
      view.setUint16(6, 0, true);          // General purpose bit flag
      view.setUint16(8, 0, true);          // Compression method (0 = store)
      view.setUint16(10, 0, true);         // Last mod time
      view.setUint16(12, 0, true);         // Last mod date
      view.setUint32(14, crc, true);
      view.setUint32(18, size, true);      // Compressed size
      view.setUint32(22, size, true);      // Uncompressed size
      view.setUint16(26, nameBytes.length, true);
      view.setUint16(28, 0, true);         // Extra field length
      localHeader.set(nameBytes, 30);

      parts.push(localHeader);
      parts.push(dataBytes);

      fileEntries.push({
        nameBytes,
        crc,
        size,
        offset
      });

      offset += localHeader.length + size;
    }

    // Central directory headers
    const centralDirStart = offset;
    let centralDirSize = 0;

    for (const entry of fileEntries) {
      const cdHeader = new Uint8Array(46 + entry.nameBytes.length);
      const view = new DataView(cdHeader.buffer);
      view.setUint32(0, 0x02014b50, true); // Central header signature
      view.setUint16(4, 20, true);         // Version made by
      view.setUint16(6, 20, true);         // Version needed
      view.setUint16(8, 0, true);          // Flag
      view.setUint16(10, 0, true);         // Compression (0)
      view.setUint16(12, 0, true);         // Mod time
      view.setUint16(14, 0, true);         // Mod date
      view.setUint32(16, entry.crc, true);
      view.setUint32(20, entry.size, true);
      view.setUint32(24, entry.size, true);
      view.setUint16(28, entry.nameBytes.length, true);
      view.setUint16(30, 0, true);
      view.setUint16(32, 0, true);
      view.setUint16(34, 0, true);
      view.setUint16(36, 0, true);
      view.setUint32(38, 0, true);
      view.setUint32(42, entry.offset, true);
      cdHeader.set(entry.nameBytes, 46);

      parts.push(cdHeader);
      centralDirSize += cdHeader.length;
    }

    // End of central directory record (22 bytes)
    const eocd = new Uint8Array(22);
    const eocdView = new DataView(eocd.buffer);
    eocdView.setUint32(0, 0x06054b50, true);
    eocdView.setUint16(4, 0, true);
    eocdView.setUint16(6, 0, true);
    eocdView.setUint16(8, fileEntries.length, true);
    eocdView.setUint16(10, fileEntries.length, true);
    eocdView.setUint32(12, centralDirSize, true);
    eocdView.setUint32(16, centralDirStart, true);
    eocdView.setUint16(20, 0, true);
    parts.push(eocd);

    return new Blob(parts, { type: 'application/zip' });
  }

  async function uploadFilesOptimized(files, labelHint = 'Uploaded Chapter') {
    if (!files || !files.length) return;

    if (pageCountBadge) pageCountBadge.textContent = 'Packaging...';
    if (loadedFolderLabel) loadedFolderLabel.textContent = `Packaging ${files.length} pages into zip...`;

    try {
      // 1. If it's already a single zip/cbz file, upload directly
      if (files.length === 1 && (files[0].name.endsWith('.zip') || files[0].name.endsWith('.cbz'))) {
        const formData = new FormData();
        formData.append('file', files[0]);
        const res = await fetch('/api/upload-zip', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
          currentPages = data.images || [];
          renderThumbnails(data.images, data.chapter_name);
        } else {
          alert(data.detail || 'Failed to extract archive.');
        }
        return;
      }

      // Filter valid image files
      const validImages = [];
      for (const file of files) {
        if (file.type.startsWith('image/') || file.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i)) {
          validImages.push(file);
        }
      }

      if (validImages.length === 0) {
        alert('No supported manga scan image files found.');
        return;
      }

      // 2. Ultra-fast zip bundle in memory (takes ~15ms)
      const zipBlob = await packFilesToZip(validImages);
      const zipFile = new File([zipBlob], `${labelHint.replace(/[^\w\-_.]/g, '_')}.zip`, { type: 'application/zip' });

      const formData = new FormData();
      formData.append('file', zipFile);

      if (loadedFolderLabel) loadedFolderLabel.textContent = `Uploading single archive (${(zipBlob.size / (1024*1024)).toFixed(1)} MB)...`;

      const res = await fetch('/api/upload-zip', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.status === 'success') {
        currentPages = data.images || [];
        renderThumbnails(data.images, data.chapter_name || labelHint);
      } else {
        alert(data.detail || 'Upload failed.');
      }
    } catch (err) {
      console.error('Fast zip upload error:', err);
      alert('Error packaging and uploading files: ' + err.message);
    }
  }

  // Choose Folder Button (triggers instant native Windows folder picker)
  if (selectFolderBtn) {
    selectFolderBtn.addEventListener('click', () => {
      triggerBackendFolderPicker();
    });
  }

  async function triggerBackendFolderPicker() {
    try {
      if (selectFolderBtn) {
        selectFolderBtn.disabled = true;
        selectFolderBtn.textContent = 'Selecting Folder...';
      }
      const res = await fetch('/api/select-folder', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        currentPages = data.images || [];
        renderThumbnails(data.images, data.chapter_name);
      }
    } catch (err) {
      console.error('Folder selection failed:', err);
    } finally {
      if (selectFolderBtn) {
        selectFolderBtn.disabled = false;
        selectFolderBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>
          Select Chapter Folder
        `;
      }
    }
  }

  // Handle Individual File / Zip Picker Button
  if (selectFilesBtn && fileInput) {
    selectFilesBtn.addEventListener('click', () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        uploadFilesOptimized(Array.from(e.target.files), 'Uploaded Chapter');
      }
    });
  }

  // Handle Browser Folder Input
  if (folderInput) {
    folderInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        let folderName = 'Uploaded Chapter';
        for (const file of e.target.files) {
          if (file.webkitRelativePath) {
            const parts = file.webkitRelativePath.split('/');
            if (parts.length > 1) {
              folderName = parts[0];
              break;
            }
          }
        }
        uploadFilesOptimized(Array.from(e.target.files), folderName);
      }
    });
  }

  // Drag and drop dropzone
  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        uploadFilesOptimized(Array.from(e.dataTransfer.files), 'Dropped Chapter');
      }
    });
  }

  if (clearPagesBtn) {
    clearPagesBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/clear-files', { method: 'POST' });
        currentPages = [];
        renderThumbnails([], '');
      } catch (err) {
        console.error('Clear files error:', err);
      }
    });
  }

  // ==========================================
  // 2. LORA MANAGEMENT & DYNAMIC STACK
  // ==========================================

  async function loadLoras() {
    try {
      const res = await fetch('/api/loras');
      const data = await res.json();
      installedLoras = data.loras || [];
      renderLoraStack();
    } catch (err) {
      console.error('Failed to fetch LoRAs:', err);
    }
  }

  function renderLoraStack() {
    if (!loraStack) return;
    if (installedLoras.length === 0) {
      loraStack.innerHTML = '<div class="loading-placeholder">No LoRAs configured. Click "Add LoRA" above.</div>';
      return;
    }

    loraStack.innerHTML = installedLoras.map((lora, idx) => {
      const tagHtml = lora.is_turbo 
        ? '<span class="lora-tag lora-tag-turbo">Turbo</span>'
        : (lora.is_custom ? '<span class="lora-tag lora-tag-custom">Custom</span>' : '');
      const deleteBtnHtml = lora.is_custom 
        ? `<button type="button" class="btn-delete-lora" data-id="${lora.id}" title="Remove LoRA">&times;</button>` 
        : '';

      return `
        <div class="lora-item ${lora.enabled ? 'active' : ''}" id="loraItem_${lora.id}">
          <div class="lora-top">
            <label class="toggle-switch">
              <input type="checkbox" class="lora-toggle-input" data-index="${idx}" ${lora.enabled ? 'checked' : ''}>
              <span class="slider"></span>
            </label>
            <div class="lora-meta">
              <div class="lora-name-row">
                <span class="lora-name">${escapeHtml(lora.name)}</span>
                ${tagHtml}
              </div>
              <div class="lora-desc" title="${escapeHtml(lora.path || '')}">${escapeHtml(lora.description || lora.filename || lora.path || '')}</div>
            </div>
            <div class="lora-val-badge" id="loraBadge_${lora.id}">${parseFloat(lora.scale).toFixed(2)}</div>
            ${deleteBtnHtml}
          </div>
          <input type="range" class="range-slider lora-slider-input" data-index="${idx}" min="0.0" max="1.5" step="0.05" value="${lora.scale}">
        </div>
      `;
    }).join('');

    // Attach Toggle Listeners
    loraStack.querySelectorAll('.lora-toggle-input').forEach(input => {
      input.addEventListener('change', (e) => {
        const idx = parseInt(e.target.dataset.index);
        const lora = installedLoras[idx];
        if (lora) {
          lora.enabled = e.target.checked;
          const parentItem = document.getElementById(`loraItem_${lora.id}`);
          if (parentItem) {
            parentItem.classList.toggle('active', lora.enabled);
          }
          fetch('/api/loras/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: lora.id, enabled: lora.enabled })
          }).catch(console.error);
        }
      });
    });

    // Attach Range Slider Listeners
    loraStack.querySelectorAll('.lora-slider-input').forEach(slider => {
      slider.addEventListener('input', (e) => {
        const idx = parseInt(e.target.dataset.index);
        const lora = installedLoras[idx];
        if (lora) {
          lora.scale = parseFloat(e.target.value);
          const badge = document.getElementById(`loraBadge_${lora.id}`);
          if (badge) {
            badge.textContent = lora.scale.toFixed(2);
          }
        }
      });

      slider.addEventListener('change', (e) => {
        const idx = parseInt(e.target.dataset.index);
        const lora = installedLoras[idx];
        if (lora) {
          fetch('/api/loras/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: lora.id, scale: lora.scale })
          }).catch(console.error);
        }
      });
    });

    // Attach Delete Listeners
    loraStack.querySelectorAll('.btn-delete-lora').forEach(btn => {
      btn.addEventListener('click', async () => {
        const loraId = btn.dataset.id;
        if (!confirm('Are you sure you want to remove this LoRA?')) return;
        try {
          const res = await fetch(`/api/loras/${loraId}`, { method: 'DELETE' });
          if (res.ok) {
            installedLoras = installedLoras.filter(l => l.id !== loraId);
            renderLoraStack();
          }
        } catch (err) {
          console.error('Delete failed:', err);
        }
      });
    });
  }

  // Add LoRA Tabs (By Link vs Upload from PC)
  if (tabLoraLink && tabLoraUpload) {
    tabLoraLink.addEventListener('click', () => {
      tabLoraLink.classList.add('active');
      tabLoraUpload.classList.remove('active');
      if (paneLoraLink) paneLoraLink.style.display = 'block';
      if (paneLoraUpload) paneLoraUpload.style.display = 'none';
      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
    });

    tabLoraUpload.addEventListener('click', () => {
      tabLoraUpload.classList.add('active');
      tabLoraLink.classList.remove('active');
      if (paneLoraUpload) paneLoraUpload.style.display = 'block';
      if (paneLoraLink) paneLoraLink.style.display = 'none';
      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
    });
  }

  // File Upload Box for LoRA .safetensors
  if (loraFileUploadBox && loraFileInput) {
    loraFileUploadBox.addEventListener('click', () => loraFileInput.click());

    loraFileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        selectedLoraFile = e.target.files[0];
        if (loraFileNameLabel) loraFileNameLabel.textContent = selectedLoraFile.name;
        loraFileUploadBox.style.borderColor = 'var(--accent-cyan)';
      }
    });

    loraFileUploadBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      loraFileUploadBox.style.borderColor = 'var(--accent-cyan)';
      loraFileUploadBox.style.background = 'rgba(0, 240, 255, 0.05)';
    });

    loraFileUploadBox.addEventListener('dragleave', () => {
      loraFileUploadBox.style.borderColor = 'var(--border)';
      loraFileUploadBox.style.background = 'rgba(255, 255, 255, 0.02)';
    });

    loraFileUploadBox.addEventListener('drop', (e) => {
      e.preventDefault();
      loraFileUploadBox.style.borderColor = 'var(--border)';
      loraFileUploadBox.style.background = 'rgba(255, 255, 255, 0.02)';
      if (e.dataTransfer.files.length) {
        const f = e.dataTransfer.files[0];
        if (f.name.endsWith('.safetensors')) {
          selectedLoraFile = f;
          if (loraFileNameLabel) loraFileNameLabel.textContent = f.name;
          loraFileUploadBox.style.borderColor = 'var(--accent-cyan)';
        } else {
          alert('Please drop a .safetensors file.');
        }
      }
    });
  }

  // Toggle & Close Add LoRA Modal
  if (toggleAddLoraBtn) {
    toggleAddLoraBtn.addEventListener('click', () => {
      if (!addLoraForm) return;
      const isHidden = addLoraForm.style.display === 'none';
      addLoraForm.style.display = isHidden ? 'block' : 'none';
      if (isHidden && newLoraUrl) newLoraUrl.focus();
    });
  }

  if (closeAddLoraBtn) {
    closeAddLoraBtn.addEventListener('click', () => {
      if (addLoraForm) addLoraForm.style.display = 'none';
      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
    });
  }

  if (cancelAddLoraBtn) {
    cancelAddLoraBtn.addEventListener('click', () => {
      if (addLoraForm) addLoraForm.style.display = 'none';
      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
    });
  }

  if (cancelUploadLoraBtn) {
    cancelUploadLoraBtn.addEventListener('click', () => {
      if (addLoraForm) addLoraForm.style.display = 'none';
      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
    });
  }

  // Submit via Link
  if (submitAddLoraBtn) {
    submitAddLoraBtn.addEventListener('click', async () => {
      const url = newLoraUrl ? newLoraUrl.value.trim() : '';
      const name = newLoraName ? newLoraName.value.trim() : '';
      const scale = newLoraScale ? parseFloat(newLoraScale.value) || 1.0 : 1.0;
      const isTurbo = newLoraIsTurbo ? newLoraIsTurbo.checked : false;
      const civitaiToken = newLoraCivitaiToken ? newLoraCivitaiToken.value.trim() : '';

      if (!url) {
        if (addLoraErrorMsg) {
          addLoraErrorMsg.textContent = 'Please enter a download link or Hugging Face path.';
          addLoraErrorMsg.style.display = 'block';
        }
        return;
      }

      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
      const btnText = submitAddLoraBtn.querySelector('.btn-text');
      const btnSpinner = submitAddLoraBtn.querySelector('.btn-spinner');
      submitAddLoraBtn.disabled = true;
      if (btnText) btnText.textContent = 'Downloading...';
      if (btnSpinner) btnSpinner.style.display = 'inline-block';

      try {
        const res = await fetch('/api/loras/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url,
            name: name,
            scale: scale,
            is_turbo: isTurbo,
            civitai_token: civitaiToken
          })
        });

        const data = await res.json();
        if (res.ok && data.status === 'success') {
          if (newLoraUrl) newLoraUrl.value = '';
          if (newLoraName) newLoraName.value = '';
          if (newLoraScale) newLoraScale.value = '1.0';
          if (newLoraIsTurbo) newLoraIsTurbo.checked = false;
          if (newLoraCivitaiToken) newLoraCivitaiToken.value = '';
          if (addLoraForm) addLoraForm.style.display = 'none';
          await loadLoras();
        } else {
          if (addLoraErrorMsg) {
            addLoraErrorMsg.textContent = data.detail || 'Failed to download LoRA.';
            addLoraErrorMsg.style.display = 'block';
          }
        }
      } catch (err) {
        if (addLoraErrorMsg) {
          addLoraErrorMsg.textContent = 'Network or server error occurred while adding LoRA.';
          addLoraErrorMsg.style.display = 'block';
        }
      } finally {
        submitAddLoraBtn.disabled = false;
        if (btnText) btnText.textContent = 'Download & Add';
        if (btnSpinner) btnSpinner.style.display = 'none';
      }
    });
  }

  // Submit via File Upload
  if (submitUploadLoraBtn) {
    submitUploadLoraBtn.addEventListener('click', async () => {
      if (!selectedLoraFile) {
        if (addLoraErrorMsg) {
          addLoraErrorMsg.textContent = 'Please select a .safetensors file from your PC first.';
          addLoraErrorMsg.style.display = 'block';
        }
        return;
      }

      if (addLoraErrorMsg) addLoraErrorMsg.style.display = 'none';
      const btnText = submitUploadLoraBtn.querySelector('.btn-text');
      const btnSpinner = submitUploadLoraBtn.querySelector('.btn-spinner');
      submitUploadLoraBtn.disabled = true;
      if (btnText) btnText.textContent = 'Uploading...';
      if (btnSpinner) btnSpinner.style.display = 'inline-block';

      try {
        const formData = new FormData();
        formData.append('file', selectedLoraFile);
        formData.append('name', uploadLoraName ? uploadLoraName.value.trim() : '');
        formData.append('scale', uploadLoraScale ? parseFloat(uploadLoraScale.value) || 1.0 : 1.0);
        formData.append('is_turbo', uploadLoraIsTurbo ? uploadLoraIsTurbo.checked : false);

        const res = await fetch('/api/loras/upload', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (res.ok && data.status === 'success') {
          selectedLoraFile = null;
          if (loraFileNameLabel) loraFileNameLabel.textContent = 'Click or Drop .safetensors LoRA here';
          if (loraFileUploadBox) loraFileUploadBox.style.borderColor = 'var(--border)';
          if (uploadLoraName) uploadLoraName.value = '';
          if (uploadLoraScale) uploadLoraScale.value = '1.0';
          if (uploadLoraIsTurbo) uploadLoraIsTurbo.checked = false;
          if (addLoraForm) addLoraForm.style.display = 'none';
          await loadLoras();
        } else {
          if (addLoraErrorMsg) {
            addLoraErrorMsg.textContent = data.detail || 'Failed to upload LoRA file.';
            addLoraErrorMsg.style.display = 'block';
          }
        }
      } catch (err) {
        if (addLoraErrorMsg) {
          addLoraErrorMsg.textContent = 'Network or server error occurred while uploading LoRA.';
          addLoraErrorMsg.style.display = 'block';
        }
      } finally {
        submitUploadLoraBtn.disabled = false;
        if (btnText) btnText.textContent = 'Upload & Add';
        if (btnSpinner) btnSpinner.style.display = 'none';
      }
    });
  }

  // ==========================================
  // 3. ZERO-WAIT DATASET CACHING STATUS & SYNC
  // ==========================================

  async function checkDatasetStatus() {
    try {
      const res = await fetch('/api/datasets/status');
      const data = await res.json();
      if (lorasDatasetPill) {
        if (data.loras && data.loras.exists) {
          lorasDatasetPill.className = 'pill pill-success';
          lorasDatasetPill.textContent = 'LoRAs: Synced (0-Wait)';
        } else {
          lorasDatasetPill.className = 'pill pill-warning';
          lorasDatasetPill.textContent = 'LoRAs: Needs Sync';
        }
      }
      if (modelsDatasetPill) {
        if (data.models && data.models.exists) {
          modelsDatasetPill.className = 'pill pill-success';
          modelsDatasetPill.textContent = 'Model: Synced (0-Wait)';
        } else {
          modelsDatasetPill.className = 'pill pill-muted';
          modelsDatasetPill.textContent = 'Model: Ready to Sync';
        }
      }
    } catch (err) {
      console.error('Dataset status check error:', err);
    }
  }

  if (syncLorasBtn) {
    syncLorasBtn.addEventListener('click', async () => {
      try {
        syncLorasBtn.disabled = true;
        if (lorasDatasetPill) {
          lorasDatasetPill.className = 'pill pill-warning';
          lorasDatasetPill.textContent = 'LoRAs: Syncing...';
        }
        await fetch('/api/datasets/sync-loras', { method: 'POST' });
        alert('LoRA dataset synchronization started in background! Check execution log.');
      } catch (err) {
        alert('Failed to trigger LoRA dataset sync.');
      } finally {
        syncLorasBtn.disabled = false;
      }
    });
  }

  if (syncModelsBtn) {
    syncModelsBtn.addEventListener('click', async () => {
      try {
        syncModelsBtn.disabled = true;
        if (modelsDatasetPill) {
          modelsDatasetPill.className = 'pill pill-warning';
          modelsDatasetPill.textContent = 'Model: Uploading...';
        }
        await fetch('/api/datasets/sync-models', { method: 'POST' });
        alert('Anima Base model dataset synchronization started in background! Check execution log.');
      } catch (err) {
        alert('Failed to trigger Model dataset sync.');
      } finally {
        syncModelsBtn.disabled = false;
      }
    });
  }

  // ==========================================
  // 3.5. KAGGLE & API SETTINGS MODAL
  // ==========================================

  const openSettingsBtn = document.getElementById('openSettingsBtn');
  const settingsModal = document.getElementById('settingsModal');
  const closeSettingsModalBtn = document.getElementById('closeSettingsModalBtn');
  const cancelSettingsBtn = document.getElementById('cancelSettingsBtn');
  const saveSettingsBtn = document.getElementById('saveSettingsBtn');
  const kaggleStatusText = document.getElementById('kaggleStatusText');
  const modalKaggleDot = document.getElementById('modalKaggleDot');
  const modalKaggleStatusText = document.getElementById('modalKaggleStatusText');
  const modalHfStatusBadge = document.getElementById('modalHfStatusBadge');
  const tabSettingsUpload = document.getElementById('tabSettingsUpload');
  const tabSettingsManual = document.getElementById('tabSettingsManual');
  const paneSettingsUpload = document.getElementById('paneSettingsUpload');
  const paneSettingsManual = document.getElementById('paneSettingsManual');
  const kaggleJsonDropzone = document.getElementById('kaggleJsonDropzone');
  const kaggleJsonFileInput = document.getElementById('kaggleJsonFileInput');
  const kaggleJsonFilenameLabel = document.getElementById('kaggleJsonFilenameLabel');
  const inputKaggleUsername = document.getElementById('inputKaggleUsername');
  const inputKaggleKey = document.getElementById('inputKaggleKey');
  const inputHfToken = document.getElementById('inputHfToken');
  const settingsFeedbackMsg = document.getElementById('settingsFeedbackMsg');

  function openSettings() {
    if (settingsModal) settingsModal.style.display = 'flex';
    if (settingsFeedbackMsg) settingsFeedbackMsg.style.display = 'none';
    checkSettings(false);
  }

  function closeSettings() {
    if (settingsModal) settingsModal.style.display = 'none';
  }

  if (openSettingsBtn) openSettingsBtn.addEventListener('click', openSettings);
  if (closeSettingsModalBtn) closeSettingsModalBtn.addEventListener('click', closeSettings);
  if (cancelSettingsBtn) cancelSettingsBtn.addEventListener('click', closeSettings);

  if (tabSettingsUpload && tabSettingsManual) {
    tabSettingsUpload.addEventListener('click', () => {
      tabSettingsUpload.classList.add('active');
      tabSettingsManual.classList.remove('active');
      if (paneSettingsUpload) paneSettingsUpload.style.display = 'block';
      if (paneSettingsManual) paneSettingsManual.style.display = 'none';
    });
    tabSettingsManual.addEventListener('click', () => {
      tabSettingsManual.classList.add('active');
      tabSettingsUpload.classList.remove('active');
      if (paneSettingsManual) paneSettingsManual.style.display = 'block';
      if (paneSettingsUpload) paneSettingsUpload.style.display = 'none';
    });
  }

  async function checkSettings(openIfUnauthenticated = false) {
    try {
      const res = await fetch('/api/settings');
      const data = await res.json();
      
      if (data.kaggle_authenticated && data.kaggle_username) {
        if (kaggleStatusText) kaggleStatusText.textContent = `@${data.kaggle_username}`;
        if (modalKaggleDot) {
          modalKaggleDot.className = 'status-dot';
          modalKaggleDot.style.background = 'var(--success)';
        }
        if (modalKaggleStatusText) {
          modalKaggleStatusText.textContent = `Connected as @${data.kaggle_username}`;
          modalKaggleStatusText.style.color = 'var(--success)';
        }
        if (inputKaggleUsername && !inputKaggleUsername.value) {
          inputKaggleUsername.value = data.kaggle_username;
        }
      } else {
        if (kaggleStatusText) kaggleStatusText.textContent = 'Connect Kaggle';
        if (modalKaggleDot) {
          modalKaggleDot.className = 'status-dot';
          modalKaggleDot.style.background = 'var(--danger)';
        }
        if (modalKaggleStatusText) {
          modalKaggleStatusText.textContent = 'Not Connected - Enter Kaggle Token';
          modalKaggleStatusText.style.color = 'var(--danger)';
        }
        if (openIfUnauthenticated && settingsModal) {
          settingsModal.style.display = 'flex';
        }
      }

      if (modalHfStatusBadge) {
        if (data.hf_token_configured) {
          modalHfStatusBadge.textContent = data.hf_token_preview ? `HF: ${data.hf_token_preview}` : 'HF: Set';
          modalHfStatusBadge.style.color = 'var(--accent-cyan)';
        } else {
          modalHfStatusBadge.textContent = 'HF: Not Set';
          modalHfStatusBadge.style.color = 'var(--text-dim)';
        }
      }
    } catch (err) {
      console.error('Settings check error:', err);
    }
  }

  // Upload kaggle.json handler
  if (kaggleJsonDropzone && kaggleJsonFileInput) {
    kaggleJsonDropzone.addEventListener('click', () => kaggleJsonFileInput.click());
    
    kaggleJsonDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      kaggleJsonDropzone.style.borderColor = 'var(--accent-cyan)';
    });
    kaggleJsonDropzone.addEventListener('dragleave', () => {
      kaggleJsonDropzone.style.borderColor = 'var(--border-active)';
    });
    kaggleJsonDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      kaggleJsonDropzone.style.borderColor = 'var(--border-active)';
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleKaggleJsonFile(e.dataTransfer.files[0]);
      }
    });

    kaggleJsonFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleKaggleJsonFile(e.target.files[0]);
      }
    });
  }

  async function handleKaggleJsonFile(file) {
    if (!file.name.endsWith('.json')) {
      showSettingsFeedback('Please select a valid kaggle.json file.', false);
      return;
    }
    if (kaggleJsonFilenameLabel) kaggleJsonFilenameLabel.textContent = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/settings/upload-kaggle-json', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        if (kaggleJsonFilenameLabel) kaggleJsonFilenameLabel.textContent = `✓ Connected: @${data.kaggle_username}`;
        showSettingsFeedback(`Successfully authenticated as @${data.kaggle_username}!`, true);
        checkSettings(false);
      } else {
        showSettingsFeedback(data.detail || data.message || 'Failed to authenticate kaggle.json', false);
      }
    } catch (err) {
      showSettingsFeedback(`Upload error: ${err.message}`, false);
    }
  }

  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener('click', async () => {
      const username = inputKaggleUsername ? inputKaggleUsername.value.trim() : '';
      const key = inputKaggleKey ? inputKaggleKey.value.trim() : '';
      const hfToken = inputHfToken ? inputHfToken.value.trim() : '';

      saveSettingsBtn.disabled = true;
      saveSettingsBtn.textContent = 'Verifying...';

      try {
        const res = await fetch('/api/settings/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            kaggle_username: username || null,
            kaggle_key: key || null,
            hf_token: hfToken || null
          })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
          showSettingsFeedback(data.message || 'Settings saved and verified!', true);
          checkSettings(false);
          setTimeout(closeSettings, 1200);
        } else {
          showSettingsFeedback(data.message || data.error || 'Failed to verify connection', false);
        }
      } catch (err) {
        showSettingsFeedback(`Save error: ${err.message}`, false);
      } finally {
        saveSettingsBtn.disabled = false;
        saveSettingsBtn.textContent = 'Save & Test Connection';
      }
    });
  }

  function showSettingsFeedback(msg, isSuccess) {
    if (!settingsFeedbackMsg) return;
    settingsFeedbackMsg.style.display = 'block';
    settingsFeedbackMsg.textContent = msg;
    settingsFeedbackMsg.style.background = isSuccess ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
    settingsFeedbackMsg.style.color = isSuccess ? 'var(--success)' : 'var(--danger)';
    settingsFeedbackMsg.style.border = `1px solid ${isSuccess ? 'var(--success)' : 'var(--danger)'}`;
  }

  // ==========================================
  // 4. PROMPTS & STYLE PRESETS
  // ==========================================

  const PROMPT_PRESETS = {
    default: "masterpiece, best quality, anime style, highly detailed coloring, vibrant color harmony, crisp lineart",
    shonen: "masterpiece, best quality, dynamic action anime, highly saturated vibrant shonen colors, crisp bold lineart, detailed illumination, colorful dramatic shading",
    pastel: "masterpiece, best quality, soft pastel anime colors, shoujo aesthetic, gentle warm lighting, dreamy atmosphere, subtle watercolor shading, clean lineart",
    cinematic: "masterpiece, best quality, cinematic dark atmosphere, dramatic moody lighting, deep contrast shadows, seinen aesthetic, hyperdetailed color depth, clean crisp lineart",
    chisa: ", chisa kotegawa, brown hair, brown eyes, short hair"
  };

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetKey = chip.dataset.preset;
      if (!positivePromptInput) return;
      if (presetKey === 'chisa') {
        const cur = positivePromptInput.value.trim();
        if (!cur.includes('chisa kotegawa')) {
          positivePromptInput.value = cur ? `${cur}${PROMPT_PRESETS.chisa}` : PROMPT_PRESETS.chisa.slice(2);
        }
      } else if (PROMPT_PRESETS[presetKey]) {
        presetChips.forEach(c => {
          if (c.dataset.preset !== 'chisa') c.classList.remove('active');
        });
        chip.classList.add('active');
        positivePromptInput.value = PROMPT_PRESETS[presetKey];
      }
    });
  });

  if (resetPromptBtn) {
    resetPromptBtn.addEventListener('click', () => {
      if (positivePromptInput) positivePromptInput.value = PROMPT_PRESETS.default;
      if (negativePromptInput) negativePromptInput.value = "worst quality, low quality, blurry, artifact, oversaturated, deformed, greyscale, monochrome";
      if (guidanceScaleSlider) {
        guidanceScaleSlider.value = "1.0";
        if (guidanceScaleVal) guidanceScaleVal.textContent = "1.0";
      }
      presetChips.forEach(c => {
        c.classList.toggle('active', c.dataset.preset === 'default');
      });
    });
  }

  if (guidanceScaleSlider && guidanceScaleVal) {
    guidanceScaleSlider.addEventListener('input', (e) => {
      guidanceScaleVal.textContent = parseFloat(e.target.value).toFixed(1);
    });
  }

  // ==========================================
  // 5. START COLORIZATION & POLLING
  // ==========================================

  if (startColorizeBtn) {
    startColorizeBtn.addEventListener('click', async () => {
      const lorasPayload = installedLoras.map(l => ({
        name: l.name,
        path: l.path || l.filename,
        scale: parseFloat(l.scale),
        enabled: !!l.enabled,
        is_turbo: !!l.is_turbo
      }));

      const resolutionSelect = document.getElementById('resolutionSelect');
      const stepsSelect = document.getElementById('stepsSelect');

      const payload = {
        chapter_name: (loadedFolderLabel && loadedFolderLabel.textContent.split(' (')[0]) || 'chapter_01',
        max_dim: resolutionSelect ? parseInt(resolutionSelect.value) : 1440,
        num_steps: stepsSelect ? parseInt(stepsSelect.value) : 8,
        guidance_scale: guidanceScaleSlider ? parseFloat(guidanceScaleSlider.value) : 1.0,
        prompt: positivePromptInput ? positivePromptInput.value.trim() : '',
        negative_prompt: negativePromptInput ? negativePromptInput.value.trim() : '',
        loras: lorasPayload
      };

      try {
        startColorizeBtn.disabled = true;
        startColorizeBtn.innerHTML = 'Starting TPU Cluster...';
        const res = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
          startPolling();
        } else {
          alert(data.detail || 'Failed to start job');
          startColorizeBtn.disabled = false;
          startColorizeBtn.textContent = 'Colorize Chapter on 8-Core TPU';
        }
      } catch (err) {
        console.error(err);
        startColorizeBtn.disabled = false;
      }
    });
  }

  function startPolling() {
    if (isPolling) return;
    isPolling = true;
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/status');
        const state = await res.json();

        // Update timer
        if (jobTimer) {
          const mins = String(Math.floor(state.elapsed_seconds / 60)).padStart(2, '0');
          const secs = String(state.elapsed_seconds % 60).padStart(2, '0');
          jobTimer.textContent = `${mins}:${secs}`;
        }
        if (phaseMessage) phaseMessage.textContent = state.phase_message;

        // Multi-Phase Visual Stepper Updater
        const setStepState = (activeId, completedIds = []) => {
          ['stepPrep', 'stepPush', 'stepCompile', 'stepColorize', 'stepExport'].forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.remove('active', 'completed');
            if (completedIds.includes(id)) el.classList.add('completed');
            if (id === activeId) el.classList.add('active');
          });
        };

        const setCoresActive = (isActive) => {
          for (let i = 0; i < 8; i++) {
            const core = document.getElementById(`tpuCore${i}`);
            if (core) {
              core.classList.toggle('active', isActive);
            }
          }
        };

        // Status styling & phase transitions
        if (state.status === 'preparing' || state.status === 'pushing') {
          if (jobStatusTitle) jobStatusTitle.textContent = 'Packaging & Pushing to Kaggle...';
          if (jobPulse) jobPulse.className = 'status-dot';
          if (progressBarFill) progressBarFill.style.width = '25%';
          if (progressPercent) progressPercent.textContent = '25%';
          setStepState('stepPush', ['stepPrep']);
          setCoresActive(false);
        } else if (state.status === 'queued') {
          if (jobStatusTitle) jobStatusTitle.textContent = 'Allocating TPU Cluster & Compiling XLA...';
          if (jobPulse) jobPulse.className = 'status-dot';
          if (progressBarFill) progressBarFill.style.width = '45%';
          if (progressPercent) progressPercent.textContent = '45%';
          setStepState('stepCompile', ['stepPrep', 'stepPush']);
          setCoresActive(false);
        } else if (state.status === 'running') {
          if (jobStatusTitle) jobStatusTitle.textContent = '8-Core Distributed Parallel Colorization';
          if (jobPulse) jobPulse.className = 'status-dot';
          if (progressBarFill) progressBarFill.style.width = '75%';
          if (progressPercent) progressPercent.textContent = '75%';
          setStepState('stepColorize', ['stepPrep', 'stepPush', 'stepCompile']);
          setCoresActive(true);
        } else if (state.status === 'downloading') {
          if (jobStatusTitle) jobStatusTitle.textContent = 'Stitching & Downloading Colorized Chapter...';
          if (jobPulse) jobPulse.className = 'status-dot';
          if (progressBarFill) progressBarFill.style.width = '92%';
          if (progressPercent) progressPercent.textContent = '92%';
          setStepState('stepExport', ['stepPrep', 'stepPush', 'stepCompile', 'stepColorize']);
          setCoresActive(false);
        } else if (state.status === 'completed') {
          if (jobStatusTitle) jobStatusTitle.textContent = 'Colorization Completed!';
          if (jobPulse) jobPulse.className = 'status-dot';
          if (progressBarFill) progressBarFill.style.width = '100%';
          if (progressPercent) progressPercent.textContent = '100%';
          setStepState('', ['stepPrep', 'stepPush', 'stepCompile', 'stepColorize', 'stepExport']);
          setCoresActive(false);
          clearInterval(pollInterval);
          isPolling = false;
          if (startColorizeBtn) {
            startColorizeBtn.disabled = false;
            startColorizeBtn.textContent = 'Colorize Chapter on 8-Core TPU';
          }
          loadResults();
        } else if (state.status === 'error') {
          if (jobStatusTitle) jobStatusTitle.textContent = 'Error during colorization';
          setCoresActive(false);
          clearInterval(pollInterval);
          isPolling = false;
          if (startColorizeBtn) {
            startColorizeBtn.disabled = false;
            startColorizeBtn.textContent = 'Colorize Chapter on 8-Core TPU';
          }
        }

        // Render logs
        if (terminalLog && state.logs && state.logs.length) {
          terminalLog.innerHTML = state.logs.map(line => `<div class="log-line">${escapeHtml(line)}</div>`).join('');
          terminalLog.scrollTop = terminalLog.scrollHeight;
        }

      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 4000);
  }

  // ==========================================
  // 6. GALLERY & SPLIT COMPARISON
  // ==========================================

  if (galleryChapterSelect) {
    galleryChapterSelect.addEventListener('change', () => {
      loadResults(galleryChapterSelect.value);
    });
  }

  async function loadResults(chapterName = null) {
    try {
      const url = chapterName ? `/api/results?chapter=${encodeURIComponent(chapterName)}` : '/api/results';
      const res = await fetch(url);
      const data = await res.json();

      // Populate available chapters
      if (galleryChapterSelect && data.available_chapters && data.available_chapters.length) {
        galleryChapterSelect.innerHTML = data.available_chapters.map(ch => 
          `<option value="${ch}" ${ch === data.chapter_name ? 'selected' : ''}>${ch.replace(/_/g, ' ')}</option>`
        ).join('');
      }

      resultPairs = data.pairs || [];
      if (resultPairs.length > 0) {
        renderGallery(resultPairs);
        selectComparisonPage(0);
      } else {
        if (compColorImg) {
          compColorImg.removeAttribute('src');
          compColorImg.style.display = 'none';
        }
        if (compInputImg) {
          compInputImg.removeAttribute('src');
          compInputImg.style.display = 'none';
        }
        if (sbsColorImg) {
          sbsColorImg.removeAttribute('src');
          sbsColorImg.style.display = 'none';
        }
        if (sbsInputImg) {
          sbsInputImg.removeAttribute('src');
          sbsInputImg.style.display = 'none';
        }
        if (pageCarousel) {
          pageCarousel.innerHTML = '<div style="color: var(--text-dim); font-size: 0.82rem; padding: 12px; width: 100%; text-align: center;">No colorized pages found for this chapter. Click "Colorize Chapter on 8-Core TPU" to start!</div>';
        }
        if (sbsPageCarousel) {
          sbsPageCarousel.innerHTML = '<div style="color: var(--text-dim); font-size: 0.82rem; padding: 12px; width: 100%; text-align: center;">No colorized pages found for this chapter.</div>';
        }
        if (resultsGrid) {
          resultsGrid.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem; padding: 24px; text-align: center; grid-column: 1 / -1;">No colorized pages available yet.</div>';
        }
      }
    } catch (err) {
      console.error('Failed to load results:', err);
    }
  }

  function renderGallery(pairs) {
    if (!resultsGrid) return;

    const renderThumbs = () => pairs.map((pair, idx) => `
      <div class="page-thumb ${idx === 0 ? 'active' : ''}" data-index="${idx}">
        <img src="${pair.color_url}" alt="${pair.id}">
        <span class="page-thumb-num">P${idx + 1}</span>
      </div>
    `).join('');

    // Render Carousel for Split View
    if (pageCarousel) {
      pageCarousel.innerHTML = renderThumbs();
      pageCarousel.querySelectorAll('.page-thumb').forEach(thumb => {
        thumb.addEventListener('click', () => {
          selectComparisonPage(parseInt(thumb.dataset.index));
        });
      });
    }

    // Render Carousel for Side-by-Side View
    if (sbsPageCarousel) {
      sbsPageCarousel.innerHTML = renderThumbs();
      sbsPageCarousel.querySelectorAll('.page-thumb').forEach(thumb => {
        thumb.addEventListener('click', () => {
          selectComparisonPage(parseInt(thumb.dataset.index));
        });
      });
    }

    // Render Grid View
    resultsGrid.innerHTML = pairs.map((pair, idx) => `
      <div class="grid-card">
        <img src="${pair.color_url}" alt="${pair.id}" loading="lazy">
        <div class="grid-card-footer">
          <span>Page ${idx + 1}</span>
          <a href="${pair.color_url}" download="${pair.color_name}" class="btn-text-danger" style="color: var(--primary);">Save</a>
        </div>
      </div>
    `).join('');
  }

  function selectComparisonPage(idx) {
    if (!resultPairs[idx]) return;
    activePairIndex = idx;
    const pair = resultPairs[idx];

    if (compColorImg) {
      compColorImg.src = pair.color_url;
      compColorImg.style.display = 'block';
    }
    if (compInputImg) {
      compInputImg.src = pair.input_url || pair.color_url;
      compInputImg.style.display = 'block';
    }
    if (sbsColorImg) {
      sbsColorImg.src = pair.color_url;
      sbsColorImg.style.display = 'block';
    }
    if (sbsInputImg) {
      sbsInputImg.src = pair.input_url || pair.color_url;
      sbsInputImg.style.display = 'block';
    }

    if (pageCarousel) {
      pageCarousel.querySelectorAll('.page-thumb').forEach((thumb, i) => {
        thumb.classList.toggle('active', i === idx);
      });
    }
    if (sbsPageCarousel) {
      sbsPageCarousel.querySelectorAll('.page-thumb').forEach((thumb, i) => {
        thumb.classList.toggle('active', i === idx);
      });
    }
  }

  // Interactive Comparison Split Slider
  const compRangeSlider = document.getElementById('compRangeSlider');
  let isDragging = false;

  function setSliderPercent(percent) {
    if (!comparisonSlider) return;
    const clamped = Math.max(0, Math.min(100, percent));
    comparisonSlider.style.setProperty('--slider-pos', `${clamped}%`);
    if (compRangeSlider && Math.abs(parseFloat(compRangeSlider.value) - clamped) > 0.5) {
      compRangeSlider.value = clamped;
    }
  }

  function updateSliderPosition(clientX) {
    if (!comparisonSlider) return;
    const rect = comparisonSlider.getBoundingClientRect();
    let pos = (clientX - rect.left) / rect.width;
    pos = Math.max(0, Math.min(1, pos));
    setSliderPercent(pos * 100);
  }

  if (compRangeSlider) {
    compRangeSlider.addEventListener('input', (e) => {
      setSliderPercent(parseFloat(e.target.value));
    });
  }

  if (comparisonSlider) {
    comparisonSlider.addEventListener('mousedown', (e) => {
      isDragging = true;
      updateSliderPosition(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      updateSliderPosition(e.clientX);
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // Touch support for mobile/tablets
    comparisonSlider.addEventListener('touchstart', (e) => {
      isDragging = true;
      if (e.touches && e.touches[0]) {
        updateSliderPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
      if (!isDragging) return;
      if (e.touches && e.touches[0]) {
        updateSliderPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isDragging = false;
    });
  }

  // Gallery Tabs (Split Compare vs Side-by-Side vs All Pages Grid)
  function switchGalleryTab(activeTab) {
    if (tabSplitView) tabSplitView.classList.toggle('active', activeTab === 'split');
    if (tabSideBySide) tabSideBySide.classList.toggle('active', activeTab === 'sbs');
    if (tabGridView) tabGridView.classList.toggle('active', activeTab === 'grid');

    if (splitViewContainer) splitViewContainer.style.display = activeTab === 'split' ? 'flex' : 'none';
    if (sideBySideContainer) sideBySideContainer.style.display = activeTab === 'sbs' ? 'flex' : 'none';
    if (gridViewContainer) gridViewContainer.style.display = activeTab === 'grid' ? 'block' : 'none';
  }

  if (tabSplitView) {
    tabSplitView.addEventListener('click', () => switchGalleryTab('split'));
  }
  if (tabSideBySide) {
    tabSideBySide.addEventListener('click', () => switchGalleryTab('sbs'));
  }
  if (tabGridView) {
    tabGridView.addEventListener('click', () => switchGalleryTab('grid'));
  }

  // Open output folder in explorer
  if (openOutputBtn) {
    openOutputBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/open-folder', { method: 'POST' });
      } catch (err) {
        console.error('Open folder error:', err);
      }
    });
  }

  // Helper string escaper
  function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
  }

  // Global Keyboard Shortcuts for Studio Viewer
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) {
      return;
    }

    if (e.key === 'ArrowRight' || e.key === 'd') {
      if (resultPairs.length > 0) {
        const nextIdx = (activePairIndex + 1) % resultPairs.length;
        selectComparisonPage(nextIdx);
      }
    } else if (e.key === 'ArrowLeft' || e.key === 'a') {
      if (resultPairs.length > 0) {
        const prevIdx = (activePairIndex - 1 + resultPairs.length) % resultPairs.length;
        selectComparisonPage(prevIdx);
      }
    } else if (e.key === '1') {
      switchGalleryTab('split');
    } else if (e.key === '2') {
      switchGalleryTab('sbs');
    } else if (e.key === '3') {
      switchGalleryTab('grid');
    }
  });

  // ==========================================
  // 7. INITIAL LAUNCH STARTUP
  // ==========================================
  checkSettings(false);
  loadResults();
  loadLoras();
  checkDatasetStatus();
});
