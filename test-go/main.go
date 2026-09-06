package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ================= 配置 =================
const (
	TempDirPrefix = "./temp_shards_"
	OutputFile    = "output.csv"
)

// Message 原始结构体
type Message struct {
	Sender       uint64    `json:"sender"`
	Rcvtime      float64   `json:"rcvTime"`
	Sendtime     float64   `json:"sendTime"`
	Type         uint32    `json:"type"`
	SenderPseudo uint64    `json:"senderPseudo"`
	Pos          []float64 `json:"pos"`
	Receiver     uint64    `json:"receiver"`
}

// ShardInfo 存储每个时间片的文件和锁
type ShardInfo struct {
	File *os.File
	Lock *sync.Mutex
}

// 全局变量
var (
	attackSenderSet = make(map[uint64]bool)
	shardMap        = make(map[int64]*ShardInfo) // 时间戳 -> 文件+锁
	shardMapMutex   sync.RWMutex                 // 保护 shardMap 本身的读写
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: go run main.go <input_dir> <min_time> <max_time>")
		return
	}

	inputDir := os.Args[1]
	minT, err := strconv.ParseInt(os.Args[2], 10, 64)
	if err != nil { panic("Invalid min_time") }
	maxT, err := strconv.ParseInt(os.Args[3], 10, 64)
	if err != nil { panic("Invalid max_time") }

	fmt.Printf("=== Start ===\nRange: [%d, %d]\n", minT, maxT)

	// 1. 创建临时目录
	tempDir := fmt.Sprintf("%s%d", TempDirPrefix, time.Now().UnixNano())
	os.MkdirAll(tempDir, 0755)
	defer os.RemoveAll(tempDir)

	// 2. 扫描文件 & 识别攻击者
	files, _ := scanFiles(inputDir)
	fmt.Printf("Found %d files, Attackers: %d\n", len(files), len(attackSenderSet))

	// 3. 【关键】预创建所有时间片文件和锁
	fmt.Println("Pre-creating shard files...")
	for t := minT; t <= maxT; t++ {
		path := filepath.Join(tempDir, fmt.Sprintf("tmp_%d.csv", t))
		f, err := os.Create(path)
		if err != nil { panic(err) }
		shardMap[t] = &ShardInfo{File: f, Lock: &sync.Mutex{}}
	}

	// 4. 并发读取与写入
	var wg sync.WaitGroup
	sem := make(chan struct{}, 20) // 限制并发数

	for _, fPath := range files {
		wg.Add(1)
		sem <- struct{}{}
		go func(path string) {
			defer wg.Done()
			defer func() { <-sem }()
			processFile(path, minT, maxT)
		}(fPath)
	}

	wg.Wait()
	fmt.Println("Reading done. Closing shard files...")

	// 5. 关闭所有分片文件
	for _, s := range shardMap {
		s.File.Close()
	}

	// 6. 合并文件
	fmt.Println("Merging...")
	mergeShards(tempDir, minT, maxT)
	fmt.Printf("=== Done! Output: %s ===\n", OutputFile)
}

func scanFiles(dir string) ([]string, error) {
	entries, _ := os.ReadDir(dir)
	var files []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, "traceJSON-") && strings.HasSuffix(name, ".json") {
			files = append(files, filepath.Join(dir, name))
			// 提取攻击者 ID (假设格式: traceJSON-Sender-Receiver-Type.json)
			parts := strings.Split(strings.TrimSuffix(name, ".json"), "-")
			if len(parts) >= 4 && parts[3] != "A0" {
				if id, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
					attackSenderSet[id] = true
				}
			}
		}
	}
	return files, nil
}

func processFile(filePath string, minT, maxT int64) {
	f, err := os.Open(filePath)
	if err != nil { return }
	defer f.Close()

	// 提取文件名中的 ReceiverID (备用)
	parts := strings.Split(strings.TrimSuffix(filepath.Base(filePath), ".json"), "-")
	fileRecvID, _ := strconv.ParseUint(parts[2], 10, 64)

	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 { continue }

		var msg Message
		if json.Unmarshal(line, &msg) != nil { continue }

		ts := int64(msg.Rcvtime)
		if ts < minT || ts > maxT { continue }

		if msg.Type != 3 { continue }

		// 确定 Receiver
		rID := msg.Receiver
		if rID == 0 { rID = fileRecvID }

		// 判断攻击
		isAttack := 0
		if attackSenderSet[msg.Sender] { isAttack = 1 }

		// 获取对应时间片的锁和文件
		shardMapMutex.RLock()
		shard, ok := shardMap[ts]
		shardMapMutex.RUnlock()

		if !ok { continue } // 理论上不会发生，因为已预创建

		// 【加锁写入】同一秒的数据串行化，不同秒并行
		shard.Lock.Lock()
		
		posX, posY := 0.0, 0.0
		if len(msg.Pos) >= 2 { posX, posY = msg.Pos[0], msg.Pos[1] }

		// 格式化: time,sender,pseudo,receiver,x,y,attack
		fmt.Fprintf(shard.File, "%d,%d,%d,%d,%.2f,%.2f,%d\n", 
			ts, msg.Sender, msg.SenderPseudo, rID, posX, posY, isAttack)
		
		shard.Lock.Unlock()
	}
}

func mergeShards(tempDir string, minT, maxT int64) {
	out, _ := os.Create(OutputFile)
	defer out.Close()
	w := bufio.NewWriter(out)
	
	w.WriteString("time,sender_id,sender_pseudo,receiver_id,pos_x,pos_y,attack_type\n")

	for t := minT; t <= maxT; t++ {
		path := filepath.Join(tempDir, fmt.Sprintf("tmp_%d.csv", t))
		if _, err := os.Stat(path); os.IsNotExist(err) { continue }
		
		f, _ := os.Open(path)
		bufio.NewReader(f).WriteTo(w)
		f.Close()
	}
	w.Flush()
}