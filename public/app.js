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
            resultContainer.style.display = 'none';
            
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

// 分析状態の確認
function checkAnalysisStatus(analysisId) {
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
