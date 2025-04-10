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

// MongoDB Memory Server の設定と接続
const { MongoMemoryServer } = require('mongodb-memory-server');

// MongoDB Memory Serverを起動する非同期関数
async function startServer() {
    // MongoDB Memory Serverインスタンスの作成
    const mongoServer = await MongoMemoryServer.create();
    const mongoUri = mongoServer.getUri();
    
    console.log(`MongoDB Memory Server が起動しました: ${mongoUri}`);
    
    try {
        // Mongooseで接続
        await mongoose.connect(mongoUri, {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('MongoDB Memory Server に接続しました');
        
        // アプリケーションサーバーの設定と起動
        setupAppServer();
    } catch (err) {
        console.error('MongoDB 接続エラー:', err);
        process.exit(1); // エラーコードで終了
    }
}

// MongoDBサーバーを起動
startServer().catch(err => {
    console.error('MongoDBサーバー起動エラー:', err);
    process.exit(1);
});

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
    attributes: [{  // 音楽属性の配列を追加
        name: String,
        confidence: Number
    }],
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
        const newAnalysis = new Analysis({
            analysisId: analysisId,
            fileName: req.file.originalname,
            filePath: req.file.path,
            status: 'pending'
        });
        
        await newAnalysis.save();

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
        const analysis = await Analysis.findOne({ analysisId: analysisId });
        
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
        const analysis = await Analysis.findOne({ analysisId: analysisId });
        
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
            attributes: analysis.attributes || [], // 新しい属性フィールド（存在しない場合は空配列）
            timestamp: analysis.timestamp
        });
    } catch (error) {
        console.error('分析結果取得エラー:', error);
        res.status(500).json({
            error: '分析結果取得中にエラーが発生しました'
        });
    }
});

// 分析プロセスの開始
function startAnalysisProcess(analysisId, filePath) {
    // 分析ステータスを更新
    Analysis.findOneAndUpdate(
        { analysisId: analysisId },
        { status: 'processing' }
    ).exec();

    // Pythonスクリプトを実行
    const pythonScript = path.join(__dirname, '../python/genre_classifier.py');
    
    const pythonProcess = spawn('python3', [pythonScript, filePath, analysisId]);
    
    // 標準出力からの結果取得
    pythonProcess.stdout.on('data', async (data) => {
        try {
            const result = JSON.parse(data.toString());
            
            // 分析結果を保存
            await Analysis.findOneAndUpdate(
                { analysisId: analysisId },
                {
                    status: 'completed',
                    genres: result.genres,
                    attributes: result.attributes || [] // 属性情報も保存（存在しない場合は空配列）
                }
            ).exec();
        } catch (error) {
            console.error('Pythonスクリプトの出力解析エラー:', error);
            updateAnalysisFailed(analysisId, 'Pythonスクリプトの出力解析に失敗しました');
        }
    });
    
    // エラー発生時の処理
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python処理エラー: ${data}`);
        updateAnalysisFailed(analysisId, data.toString());
    });
    
    // プロセス終了時の処理
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Pythonプロセスが異常終了しました。終了コード: ${code}`);
            updateAnalysisFailed(analysisId, `Pythonプロセスが異常終了しました（コード: ${code}）`);
        }
    });
    
    // エラー処理
    pythonProcess.on('error', (error) => {
        console.error('Pythonプロセス起動エラー:', error);
        updateAnalysisFailed(analysisId, 'Pythonプロセスの起動に失敗しました');
    });
}

// 分析失敗時の状態更新
async function updateAnalysisFailed(analysisId, errorMessage) {
    try {
        await Analysis.findOneAndUpdate(
            { analysisId: analysisId },
            {
                status: 'failed',
                error: errorMessage
            }
        ).exec();
    } catch (err) {
        console.error('分析失敗状態更新エラー:', err);
    }
}

// ルートへのアクセスでフロントエンドを提供
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

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
