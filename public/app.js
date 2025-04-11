// DOM要素
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('upload-btn');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const resultSection = document.getElementById('result-section');
const analysisLoading = document.getElementById('analysis-loading');
const resultContainer = document.getElementById('result-container');
const historyContainer = document.getElementById('history-container');
const noHistoryMessage = document.getElementById('no-history-message');
const waveformContainer = document.getElementById('waveform-container');
const waveformCanvas = document.getElementById('waveform-canvas');
const analysisDetails = document.getElementById('analysis-details');
const analysisDescription = document.getElementById('analysis-description');
const descriptionText = document.getElementById('description-text');

// ジャンル名と対応するクラス名のマッピング
const genreClasses = {
    'rock': 'rock',
    'pop': 'pop',
    'jazz': 'jazz',
    'classical': 'classical',
    'electronic': 'electronic',
    'hip-hop': 'hip-hop',
    'country': 'country',
    'blues': 'blues',
    'reggae': 'reggae',
    'metal': 'metal',
    'folk': 'pop',
    'r&b': 'blues',
    'latin': 'reggae',
    'world': 'jazz',
    'punk': 'rock',
    'funk': 'hip-hop'
    // 他のジャンルも追加可能
};

// イベントリスナーの設定
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadHistory();
});

// イベントリスナーのセットアップ
function setupEventListeners() {
    // アップロードボタンのクリックイベント
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // ファイル選択イベント
    fileInput.addEventListener('change', handleFileSelect);

    // ドラッグ＆ドロップイベント
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // ドラッグエンター・オーバーでハイライト
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    // ドラッグリーブ・ドロップでハイライト解除
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // ハイライト関数
    function highlight() {
        dropArea.classList.add('highlight');
    }

    // ハイライト解除関数
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    // ドロップイベント
    dropArea.addEventListener('drop', handleDrop, false);
}

// ドロップされたファイルの処理
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0 && files[0].type === 'audio/mpeg') {
        handleFileUpload(files[0]);
    } else {
        alert('MP3ファイルのみアップロード可能です。');
    }
}

// ファイル選択の処理
function handleFileSelect(e) {
    const files = e.target.files;
    
    if (files.length > 0 && files[0].type === 'audio/mpeg') {
        handleFileUpload(files[0]);
    } else {
        alert('MP3ファイルのみアップロード可能です。');
    }
}

// ファイルアップロード処理
function handleFileUpload(file) {
    // プログレスバー表示
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'アップロード中... 0%';
    
    // 前回の解析結果をすべて非表示にする
    clearPreviousResults();
    
    // FormDataの作成
    const formData = new FormData();
    formData.append('audioFile', file);
    
    // XHRでサーバーにアップロード
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);
    
    // アップロードの進捗イベント
    xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            progressBar.style.width = percent + '%';
            progressText.textContent = `アップロード中... ${percent}%`;
        }
    });
    
    // アップロード完了イベント
    xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
            progressText.textContent = 'アップロード完了！分析中...';
            resultSection.style.display = 'block';
            analysisLoading.style.display = 'flex';
            
            // 分析結果の取得
            const response = JSON.parse(xhr.responseText);
            const analysisId = response.analysisId;
            
            // 分析結果のポーリング
            checkAnalysisStatus(analysisId);
        } else {
            handleError('アップロードエラー: ' + xhr.statusText);
        }
    });
    
    // エラーイベント
    xhr.addEventListener('error', () => {
        handleError('ネットワークエラーが発生しました。');
    });
    
    // アップロード中止イベント
    xhr.addEventListener('abort', () => {
        handleError('アップロードが中止されました。');
    });
    
    // 送信
    xhr.send(formData);
}

// 前回の解析結果をクリアする関数
function clearPreviousResults() {
    resultContainer.style.display = 'none';
    waveformContainer.style.display = 'none';
    analysisDetails.style.display = 'none';
    analysisDescription.style.display = 'none';
}

// 分析状態の確認
function checkAnalysisStatus(analysisId) {
    // 前回の結果が表示されないようにする
    clearPreviousResults();
    
    fetch(`/api/analysis/${analysisId}/status`)
        .then(response => {
            if (!response.ok) {
                throw new Error('分析状態の取得に失敗しました');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'completed') {
                // 分析完了時、結果を取得して表示
                fetchAndDisplayResults(analysisId);
            } else if (data.status === 'failed') {
                handleError('音楽分析に失敗しました: ' + data.error);
            } else {
                // まだ分析中の場合、3秒後に再度確認
                setTimeout(() => checkAnalysisStatus(analysisId), 3000);
            }
        })
        .catch(error => {
            handleError('エラー: ' + error.message);
        });
}

// 分析結果の取得と表示
function fetchAndDisplayResults(analysisId) {
    fetch(`/api/analysis/${analysisId}/result`)
        .then(response => {
            if (!response.ok) {
                throw new Error('分析結果の取得に失敗しました');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
            addToHistory(data);
            loadHistory(); // 履歴を更新
        })
        .catch(error => {
            handleError('エラー: ' + error.message);
        });
}

// 波形データを描画
function drawWaveform(waveformData) {
    if (!waveformData || !waveformData.samples || waveformData.samples.length === 0) {
        waveformContainer.style.display = 'none';
        return;
    }
    
    waveformContainer.style.display = 'block';
    
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    
    // Canvasのサイズを設定（DPIに対応）
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    
    // 背景をクリア
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    
    const samples = waveformData.samples;
    const step = Math.ceil(samples.length / width);
    const amp = height / 2;
    
    // 中央線を描画
    ctx.beginPath();
    ctx.strokeStyle = '#ddd';
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    
    // 波形を描画
    ctx.beginPath();
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < width; i++) {
        const sampleIndex = Math.floor(i * step);
        const sample = samples[sampleIndex < samples.length ? sampleIndex : samples.length - 1];
        const y = (0.5 + sample / 2) * height; // -1~1の値を0~heightにマッピング
        
        if (i === 0) {
            ctx.moveTo(i, y);
        } else {
            ctx.lineTo(i, y);
        }
    }
    
    ctx.stroke();
}

// 詳細分析データの表示
function displayAnalysisDetails(analysisData) {
    if (!analysisData) {
        analysisDetails.style.display = 'none';
        return;
    }
    
    analysisDetails.style.display = 'block';
    
    // 分析グリッドの内容を作成
    const gridContainer = analysisDetails.querySelector('.analysis-grid');
    
    let analysisHTML = '';
    
    // カテゴリーヘッダー: 基本音楽情報
    analysisHTML += `
        <div class="analysis-category">
            <h3 class="category-header">基本音楽情報</h3>
            <div class="category-grid">
                <div class="analysis-item">
                    <div class="label">テンポ (BPM)</div>
                    <div class="value">${analysisData.tempo.toFixed(1)}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">キー</div>
                    <div class="value">${analysisData.key}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">キー確信度</div>
                    <div class="value">${analysisData.key_confidence ? (analysisData.key_confidence * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">ビート強度</div>
                    <div class="value">${analysisData.beat_strength ? (analysisData.beat_strength * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">テンポ安定性</div>
                    <div class="value">${analysisData.tempo_stability ? (analysisData.tempo_stability * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">拍子</div>
                    <div class="value">${analysisData.time_signature || '不明'}</div>
                </div>
            </div>
        </div>
    `;
    
    // カテゴリーヘッダー: 音量とダイナミクス
    analysisHTML += `
        <div class="analysis-category">
            <h3 class="category-header">音量とダイナミクス</h3>
            <div class="category-grid">
                <div class="analysis-item">
                    <div class="label">エネルギー</div>
                    <div class="value">${(analysisData.energy * 100).toFixed(0)}%</div>
                </div>
                <div class="analysis-item">
                    <div class="label">平均音量</div>
                    <div class="value">${analysisData.mean_volume ? analysisData.mean_volume.toFixed(3) : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">最大音量</div>
                    <div class="value">${analysisData.max_volume ? analysisData.max_volume.toFixed(3) : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">ダイナミックレンジ</div>
                    <div class="value">${analysisData.dynamic_range ? (analysisData.dynamic_range * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">音量変化率</div>
                    <div class="value">${analysisData.volume_change_rate ? (analysisData.volume_change_rate * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
            </div>
        </div>
    `;
    
    // カテゴリーヘッダー: スペクトル特性
    analysisHTML += `
        <div class="analysis-category">
            <h3 class="category-header">スペクトル特性</h3>
            <div class="category-grid">
                <div class="analysis-item">
                    <div class="label">スペクトル重心 (Hz)</div>
                    <div class="value">${analysisData.spectral_centroid ? Math.round(analysisData.spectral_centroid) : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">スペクトル帯域幅 (Hz)</div>
                    <div class="value">${analysisData.spectral_bandwidth ? Math.round(analysisData.spectral_bandwidth) : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">スペクトル平坦度</div>
                    <div class="value">${analysisData.spectral_flatness ? analysisData.spectral_flatness.toFixed(3) : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">音色の明るさ</div>
                    <div class="value">${analysisData.brightness ? (analysisData.brightness * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">調和性</div>
                    <div class="value">${analysisData.harmonicity ? (analysisData.harmonicity * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">音の粗さ</div>
                    <div class="value">${analysisData.roughness ? (analysisData.roughness * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
            </div>
        </div>
    `;
    
    // カテゴリーヘッダー: リズムと動き
    analysisHTML += `
        <div class="analysis-category">
            <h3 class="category-header">リズムと動き</h3>
            <div class="category-grid">
                <div class="analysis-item">
                    <div class="label">ダンス性</div>
                    <div class="value">${(analysisData.danceability * 100).toFixed(0)}%</div>
                </div>
                <div class="analysis-item">
                    <div class="label">アコースティック度</div>
                    <div class="value">${(analysisData.acousticness * 100).toFixed(0)}%</div>
                </div>
                <div class="analysis-item">
                    <div class="label">アタック強度</div>
                    <div class="value">${analysisData.attack_strength ? analysisData.attack_strength.toFixed(3) : '不明'}</div>
                </div>
            </div>
        </div>
    `;
    
    // カテゴリーヘッダー: 楽曲構造
    analysisHTML += `
        <div class="analysis-category">
            <h3 class="category-header">楽曲構造</h3>
            <div class="category-grid">
                <div class="analysis-item">
                    <div class="label">セクション数</div>
                    <div class="value">${analysisData.sections}</div>
                </div>
                <div class="analysis-item">
                    <div class="label">繰り返し度合い</div>
                    <div class="value">${analysisData.chorus_likelihood ? (analysisData.chorus_likelihood * 100).toFixed(0) + '%' : '不明'}</div>
                </div>
            </div>
        </div>
    `;
    
    // カテゴリーヘッダー: 楽器検出
    analysisHTML += `
        <div class="analysis-category">
            <h3 class="category-header">楽器検出</h3>
            <div class="instruments-chart">
    `;
    
    // 楽器ごとのバーを追加
    for (const [instrument, value] of Object.entries(analysisData.instruments)) {
        const percent = Math.round(value * 100);
        analysisHTML += `
            <div class="instrument-bar">
                <span class="instrument-name">${instrument}</span>
                <div class="instrument-level-bg">
                    <div class="instrument-level" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
    }
    
    analysisHTML += `
            </div>
        </div>
    `;
    
    gridContainer.innerHTML = analysisHTML;
}

// 総合的な説明の表示
function displayDescription(description) {
    if (!description) {
        analysisDescription.style.display = 'none';
        return;
    }
    
    analysisDescription.style.display = 'block';
    descriptionText.textContent = description;
}

// 分析結果の表示
function displayResults(result) {
    analysisLoading.style.display = 'none';
    resultContainer.style.display = 'block';
    
    // 結果カードの作成
    const resultHTML = `
        <div class="genre-card">
            <h3>分析結果</h3>
            <div class="file-info">
                <p>ファイル名: ${result.fileName}</p>
                <p>分析日時: ${new Date(result.timestamp).toLocaleString('ja-JP')}</p>
            </div>
            ${result.genres.map((genre, index) => {
                const genreClass = genreClasses[genre.name.toLowerCase()] || 'pop';
                const confidencePercent = Math.round(genre.confidence * 100);
                return `
                    <div class="genre-result">
                        <div class="genre-info">
                            <span class="genre-name">${genre.name}</span>
                            <span class="genre-confidence">${confidencePercent}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-level ${genreClass}" 
                                 style="width: ${confidencePercent}%"></div>
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
    
    resultContainer.innerHTML = resultHTML;
    
    // 波形データの描画
    if (result.waveform) {
        drawWaveform(result.waveform);
    }
    
    // 詳細分析の表示
    if (result.analysis) {
        displayAnalysisDetails(result.analysis);
    }
    
    // 総合的な説明の表示
    if (result.description) {
        displayDescription(result.description);
    }
}

// 履歴への追加
function addToHistory(result) {
    const history = JSON.parse(localStorage.getItem('genreAnalysisHistory') || '[]');
    
    // 最大保存数を制限（例: 最新10件）
    if (history.length >= 10) {
        history.pop();
    }
    
    // 先頭に追加
    history.unshift(result);
    
    // ローカルストレージに保存
    localStorage.setItem('genreAnalysisHistory', JSON.stringify(history));
}

// 履歴の読み込み
function loadHistory() {
    const history = JSON.parse(localStorage.getItem('genreAnalysisHistory') || '[]');
    
    if (history.length === 0) {
        noHistoryMessage.style.display = 'block';
        return;
    }
    
    noHistoryMessage.style.display = 'none';
    
    // 履歴の表示
    const historyHTML = history.map(item => {
        // 最も信頼度の高いジャンルを取得
        const topGenre = item.genres.reduce((prev, current) => 
            prev.confidence > current.confidence ? prev : current
        );
        
        return `
            <div class="history-item" data-id="${item.analysisId}">
                <span class="file-name">${item.fileName}</span>
                <span class="genre">${topGenre.name}</span>
                <span class="date">${new Date(item.timestamp).toLocaleString('ja-JP')}</span>
            </div>
        `;
    }).join('');
    
    historyContainer.innerHTML = historyHTML;
    
    // 履歴アイテムのクリックイベントを設定
    document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            const analysisId = item.getAttribute('data-id');
            const result = history.find(h => h.analysisId === analysisId);
            
            if (result) {
                resultSection.style.display = 'block';
                analysisLoading.style.display = 'none';
                resultContainer.style.display = 'block';
                displayResults(result);
                
                // スクロールして結果を表示
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// エラー処理
function handleError(message) {
    progressContainer.style.display = 'none';
    analysisLoading.style.display = 'none';
    
    alert(message);
    console.error(message);
}
