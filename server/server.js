const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');
const mongoose = require('mongoose');
const cors = require('cors');

// Express アプリの設定
const app = express();
const port = process.env.PORT || 3000;

// CORS設定
app.use(cors());

// JSONミドルウェア
app.use(express.json());

// 静的ファイルの提供
app.use(express.static(path.join(__dirname, '../public')));

// Mongooseの警告を抑制
mongoose.set('strictQuery', true);

// 簡易的なメモリ内データストアを作成（MongoDB代替用）
const memoryStore = {
    analyses: new Map(),
    
    async save(analysisId, data) {
        this.analyses.set(analysisId, data);
        console.log(`分析データを保存しました: ${analysisId}`);
        return data;
    },
    
    async findOne(query) {
        if (query.analysisId) {
            return this.analyses.get(query.analysisId) || null;
        }
        return null;
    },
    
    async findOneAndUpdate(query, update, options) {
        const existingData = this.analyses.get(query.analysisId) || {};
        const updatedData = { ...existingData, ...update };
        this.analyses.set(query.analysisId, updatedData);
        console.log(`分析データを更新しました: ${query.analysisId}`);
        
        // MongoDBのようにexecメソッドを持つオブジェクトを返す
        return {
            exec: function() {
                return Promise.resolve(updatedData);
            }
        };
    }
};

console.log('メモリ内データストアを初期化しました');

// サーバーを即時起動
setupAppServer();

// モジュールとしてエクスポート（他のファイルから require したときのため）
module.exports = app;

// Analysisスキーマの定義
const analysisSchema = new mongoose.Schema({
    analysisId: { 
        type: String, 
        required: true, 
        unique: true 
    },
    fileName: { 
        type: String, 
        required: true 
    },
    filePath: { 
        type: String, 
        required: true 
    },
    status: { 
        type: String, 
        enum: ['pending', 'processing', 'completed', 'failed'],
        required: true,
        default: 'pending'
    },
    genres: [{
        name: String,
        confidence: Number
    }],
    // 波形データ
    waveform: {
        samples: [Number],
        sampleRate: Number,
        duration: Number
    },
    // 詳細分析情報（拡張スキーマ）
    analysis: {
        // 基本音楽情報
        tempo: Number,
        beat_strength: Number,
        tempo_stability: Number,
        time_signature: String,
        key: String,
        key_confidence: Number,
        
        // 音量とダイナミクス
        mean_volume: Number,
        max_volume: Number,
        dynamic_range: Number,
        volume_change_rate: Number,
        energy: Number,
        
        // スペクトル特性
        spectral_centroid: Number,
        spectral_bandwidth: Number,
        spectral_rolloff: Number,
        spectral_flatness: Number,
        zero_crossing_rate: Number,
        brightness: Number,
        harmonicity: Number,
        roughness: Number,
        
        // リズムと動き
        danceability: Number,
        attack_strength: Number,
        acousticness: Number,
        
        // 構造
        sections: Number,
        chorus_likelihood: Number,
        
        // 楽器
        instruments: Object
    },
    // 総合的な音楽説明
    description: String,
    timestamp: { 
        type: Date, 
        default: Date.now 
    },
    error: String
});

// Analysisモデルの作成
const Analysis = mongoose.model('Analysis', analysisSchema);

// ファイル保存用のストレージ設定
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const uploadPath = path.join(__dirname, 'uploads');
        // アップロードディレクトリが存在しない場合は作成
        if (!fs.existsSync(uploadPath)) {
            fs.mkdirSync(uploadPath, { recursive: true });
        }
        cb(null, uploadPath);
    },
    filename: function (req, file, cb) {
        // ファイル名の衝突を避けるためにUUIDを追加
        const uniqueId = uuidv4();
        cb(null, uniqueId + '-' + file.originalname);
    }
});

// ファイルフィルター（テスト環境では拡張子で判断）
const fileFilter = (req, file, cb) => {
    // テスト環境では拡張子で判断する
    if (file.originalname.toLowerCase().endsWith('.mp3') || 
        file.mimetype === 'audio/mpeg') {
        console.log('MP3ファイルとして許可:', file.originalname);
        cb(null, true);
    } else {
        console.log('MP3以外のファイルは拒否:', file.originalname, file.mimetype);
        cb(new Error('MP3ファイルのみアップロード可能です'), false);
    }
};

// Multerの設定
const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB制限
    }
});

// ファイルアップロードエンドポイント
app.post('/api/upload', upload.single('audioFile'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'ファイルがアップロードされていません' });
        }

        const analysisId = uuidv4();
        
        // 分析レコードを作成
        const newAnalysis = {
            analysisId: analysisId,
            fileName: req.file.originalname,
            filePath: req.file.path,
            status: 'pending'
        };
        
        await memoryStore.save(analysisId, newAnalysis);

        // 分析プロセスを開始
        startAnalysisProcess(analysisId, req.file.path);
        
        res.status(200).json({
            message: 'ファイルがアップロードされました',
            analysisId: analysisId
        });
    } catch (error) {
        console.error('アップロードエラー:', error);
        res.status(500).json({
            error: 'ファイルアップロード中にエラーが発生しました'
        });
    }
});

// 分析状態取得エンドポイント
app.get('/api/analysis/:analysisId/status', async (req, res) => {
    try {
        const analysisId = req.params.analysisId;
        const analysis = await memoryStore.findOne({ analysisId: analysisId });
        
        if (!analysis) {
            return res.status(404).json({ error: '分析が見つかりません' });
        }
        
        res.status(200).json({
            status: analysis.status,
            error: analysis.error
        });
    } catch (error) {
        console.error('分析状態取得エラー:', error);
        res.status(500).json({
            error: '分析状態取得中にエラーが発生しました'
        });
    }
});

// 分析結果取得エンドポイント
app.get('/api/analysis/:analysisId/result', async (req, res) => {
    try {
        const analysisId = req.params.analysisId;
        const analysis = await memoryStore.findOne({ analysisId: analysisId });
        
        if (!analysis) {
            return res.status(404).json({ error: '分析が見つかりません' });
        }
        
        if (analysis.status !== 'completed') {
            return res.status(400).json({
                error: `分析はまだ完了していません。現在の状態: ${analysis.status}`
            });
        }
        
        res.status(200).json({
            analysisId: analysis.analysisId,
            fileName: analysis.fileName,
            genres: analysis.genres,
            waveform: analysis.waveform,
            analysis: analysis.analysis,
            description: analysis.description,
            timestamp: analysis.timestamp
        });
    } catch (error) {
        console.error('分析結果取得エラー:', error);
        res.status(500).json({
            error: '分析結果取得中にエラーが発生しました'
        });
    }
});

// ファイル削除ヘルパー関数（グローバルスコープに配置）
async function deleteUploadedFile(filePath) {
    try {
        // ファイルが存在するか確認
        if (fs.existsSync(filePath)) {
            await fs.promises.unlink(filePath);
            console.log(`ファイルを削除しました: ${filePath}`);
        } else {
            console.log(`ファイルが見つかりません: ${filePath}`);
        }
    } catch (error) {
        console.error(`ファイル削除エラー: ${filePath}`, error);
    }
}

// 分析プロセスの開始
async function startAnalysisProcess(analysisId, filePath) {
    // 分析ステータスを更新
    await memoryStore.findOneAndUpdate(
        { analysisId: analysisId },
        { status: 'processing' }
    );

    // Pythonスクリプトを実行
    const pythonScript = path.join(__dirname, '../python/genre_classifier.py');
    
    const pythonProcess = spawn('python3', [pythonScript, filePath, analysisId]);
    
    // Pythonからの出力を蓄積するバッファ
    let stdoutBuffer = '';
    
    // 標準出力からの結果取得
    pythonProcess.stdout.on('data', (data) => {
        // バッファにデータを追加
        stdoutBuffer += data.toString();
    });
    
    // プロセス終了時に完全な出力を処理
    pythonProcess.on('close', async (code) => {
        try {
            if (code === 0 && stdoutBuffer) {
                // 完全な出力が揃ったところでJSON解析を実行
                const result = JSON.parse(stdoutBuffer);
                
                // デバッグ出力
                fs.writeFileSync(
                    path.join(__dirname, '../debug_output.txt'),
                    JSON.stringify(result, null, 2)
                );
                
                // Python側から受け取ったデータを検証
                const analysisData = result.genres.analysis || {};
                console.log('受信した分析データのキー:', Object.keys(analysisData));
                
                // 分析データをより詳細にログ出力
                fs.writeFileSync(
                    path.join(__dirname, '../analysis_data.txt'),
                    JSON.stringify(analysisData, null, 2)
                );
                
                // 保存する前にデータの整形と検証
                const processedAnalysis = processAnalysisData(analysisData);
                
                // 分析結果を保存
                await memoryStore.findOneAndUpdate(
                    { analysisId: analysisId },
                    {
                        status: 'completed',
                        genres: result.genres.genres || result.genres,
                        waveform: result.genres.waveform,
                        analysis: result.genres.analysis,
                        description: result.genres.description,
                        timestamp: new Date()
                    }
                );
                
                // 分析が完了したらアップロードされたファイルを削除
                await deleteUploadedFile(filePath);
                console.log(`分析完了: ${analysisId}`);
            } else if (code !== 0) {
                console.error(`Pythonプロセスが異常終了しました。終了コード: ${code}`);
                updateAnalysisFailed(analysisId, `Pythonプロセスが異常終了しました（コード: ${code}）`);
            }
        } catch (error) {
            console.error('Pythonスクリプトの出力解析エラー:', error);
            console.error('受信したデータ（一部）:', stdoutBuffer.substring(0, 200) + '...');
            updateAnalysisFailed(analysisId, 'Pythonスクリプトの出力解析に失敗しました');
        }
    });
    
    // エラー発生時の処理
    pythonProcess.stderr.on('data', (data) => {
        const message = data.toString();
        console.error(`Python処理エラー: ${message}`);
        
        // DEBUGプレフィックスのないメッセージのみをエラーとして扱う
        if (!message.trim().startsWith('DEBUG:')) {
            updateAnalysisFailed(analysisId, message);
        }
    });
    
    // 削除: 重複したcloseイベントハンドラ（上で定義済み）
    
    // 削除：内側のファイル削除関数は削除（外側のグローバル関数を使用）
    
    // エラー処理
    pythonProcess.on('error', (error) => {
        console.error('Pythonプロセス起動エラー:', error);
        updateAnalysisFailed(analysisId, 'Pythonプロセスの起動に失敗しました');
    });
}

// 分析失敗時の状態更新
async function updateAnalysisFailed(analysisId, errorMessage) {
    try {
        const analysis = await memoryStore.findOneAndUpdate(
            { analysisId: analysisId },
            {
                status: 'failed',
                error: errorMessage
            }
        );
        
        // 分析が失敗した場合もファイルを削除
        if (analysis && analysis.filePath) {
            await deleteUploadedFile(analysis.filePath);
        }
    } catch (err) {
        console.error('分析失敗状態更新エラー:', err);
    }
}

// ルートへのアクセスでフロントエンドを提供
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

// 分析データの処理と整形
function processAnalysisData(data) {
    if (!data) return {};
    
    // 結果オブジェクト
    const processedData = {};
    
    // データの型変換と検証（数値項目）
    const numericFields = [
        'tempo', 'beat_strength', 'tempo_stability', 'key_confidence',
        'mean_volume', 'max_volume', 'dynamic_range', 'volume_change_rate', 'energy',
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness',
        'zero_crossing_rate', 'brightness', 'harmonicity', 'roughness',
        'danceability', 'attack_strength', 'acousticness',
        'sections', 'chorus_likelihood'
    ];
    
    // Python側からのデータキーを確認するログ出力
    console.log('処理中の分析データのキー一覧:', Object.keys(data));
    
    numericFields.forEach(field => {
        if (field in data) {
            // NaNやnullの場合は除外
            const value = parseFloat(data[field]);
            if (!isNaN(value) && value !== null) {
                processedData[field] = value;
            } else {
                console.log(`数値変換に失敗したフィールド: ${field}, 値: ${data[field]}`);
            }
        } else {
            processedData[field] = null; // 不明な値を明示的にnullとして保存する
        }
    });
    
    // 文字列項目
    const stringFields = ['key', 'time_signature'];
    stringFields.forEach(field => {
        if (field in data && data[field]) {
            processedData[field] = String(data[field]);
        } else {
            processedData[field] = null; // 不明な値を明示的にnullとして保存する
        }
    });
    
    // 楽器データの処理
    if (data.instruments && typeof data.instruments === 'object') {
        processedData.instruments = { ...data.instruments };
    }
    
    // Python分析結果全体をデータベースに保存するように変更
    // Pythonからのオリジナルデータをそのまま保持するが、型変換のみ行う
    Object.keys(data).forEach(key => {
        if (!processedData.hasOwnProperty(key)) {
            const value = data[key];
            if (typeof value === 'number') {
                processedData[key] = value;
            } else if (value && typeof value === 'string' && !isNaN(parseFloat(value))) {
                processedData[key] = parseFloat(value);
            } else if (value && typeof value === 'string') {
                processedData[key] = value;
            } else if (typeof value === 'object' && value !== null) {
                processedData[key] = value;
            }
        }
    });
    
    return processedData;
}

// アプリケーションサーバーのセットアップと起動関数
function setupAppServer() {
    // サーバーの起動
    app.listen(port, () => {
        console.log(`サーバーがポート ${port} で起動しました`);
        
        // アップロードディレクトリの確認
        const uploadPath = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadPath)) {
            fs.mkdirSync(uploadPath, { recursive: true });
            console.log('アップロードディレクトリを作成しました');
        }
    });
}
