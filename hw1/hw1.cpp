#include <bits/stdc++.h>
#include <boost/functional/hash.hpp>
#include <algorithm>
#include <fstream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>  // 添加線程安全的 map
#include <tbb/global_control.h>
using namespace std;

static const int DR[4]={-1,0,1,0};
static const int DC[4]={0,-1,0,1};
static const char MV[4]={'W','A','S','D'};
static const int MAXC=256;

static int PRINT_ALL_STATES = [](){ const char* v=getenv("PRINT_ALL_STATES"); return v?atoi(v):0; }();
static std::mutex g_deadlock_log_mutex; // protect concurrent logging
static std::mutex g_print_mutex; // protect PRINT_ALL_STATES printing

struct Oops:exception{
    string msg;Oops(string m):msg(std::move(m)){}
    const char* what()const noexcept override{return msg.c_str();}
};

struct LevelInfo{
    int H,W;
    int player;
    bitset<MAXC> boxes,targets;
    bitset<MAXC> is_wall,is_fragile;  // 改用 bitset
    bitset<MAXC> static_dead;         // 改用 bitset
    bitset<MAXC> corridor_dead;       // 改用 bitset
    array<bitset<MAXC>,4> tunnel_forward; // 預先計算的走廊推箱資訊
    
    // 凹槽結構：每個凹槽包含其位置和對應的入口
    struct Groove {
        vector<int> positions;    // 凹槽內的位置
        vector<int> entrances;    // 凹槽的入口位置
    };
    vector<Groove> grooves;
};
inline int idx(int r,int c,int W){return r*W+c;}
inline bool inb(int r,int c,int H,int W){return 0<=r&&r<H&&0<=c&&c<W;}

inline bool cell_allows_box(const LevelInfo& L, int pos){
    return !L.is_wall.test(pos) && !L.is_fragile.test(pos);
}

struct State{int player;bitset<MAXC> boxes;};

LevelInfo load_level(const string&fn){
    ifstream f(fn);if(!f.is_open())throw Oops("cannot open");
    vector<string> rows;string line;
    while(getline(f,line)){
        while(!line.empty()&&(line.back()=='\r'||line.back()=='\n'))line.pop_back();
        if(!line.empty()) rows.push_back(line);
    }
    if(rows.empty()) throw Oops("empty level");
    int H=rows.size(),W=rows[0].size();
    LevelInfo L;L.H=H;L.W=W;
    L.is_wall.reset();L.is_fragile.reset();  // bitset 用 reset() 清零
    bool hasPlayer=false;
    for(int r=0;r<H;r++){
        if((int)rows[r].size()!=W) throw Oops("ragged map");
        for(int c=0;c<W;c++){
            char ch=rows[r][c];int id=idx(r,c,W);
            if(ch=='#')L.is_wall.set(id);                        // bitset 用 set()
            if(ch=='@'||ch=='!')L.is_fragile.set(id);         // bitset 用 set()
            if(ch=='.'||ch=='O'||ch=='X')L.targets.set(id);
            if(ch=='x'||ch=='X')L.boxes.set(id);
            if(ch=='o'||ch=='O'||ch=='!'){L.player=id;hasPlayer=true;}
        }
    }
    if(!hasPlayer) throw Oops("no player");
    return L;
}

// ===== 印狀態（可關閉） =====
static inline void print_state(const State& s, const LevelInfo& L, const string& title=""){
    if(!title.empty()) cerr<<"=== "<<title<<" ===\n";
    for(int r=0;r<L.H;r++){
        for(int c=0;c<L.W;c++){
            int pos=idx(r,c,L.W);char ch=' ';
            if(L.is_wall.test(pos)) ch='#';              // bitset 用 test()
            else if(L.is_fragile.test(pos)) ch='@';      // bitset 用 test()
            else if(s.boxes.test(pos)&&L.targets.test(pos)) ch='X';
            else if(s.boxes.test(pos)) ch='x';
            else if(pos==s.player&&L.targets.test(pos)) ch='O';
            else if(pos==s.player) ch='o';
            else if(L.targets.test(pos)) ch='.';
            else ch=' ';
            cerr<<ch;
        } cerr<<"\n";
    }
    cerr<<"Player: ("<<s.player/L.W<<","<<s.player%L.W<<")\n\n";
}

// ===== 簡化但快速的 heuristic（貪心匹配） =====
int simple_heuristic(const State&s,const LevelInfo&L){
    vector<int> boxes, targets;
    for(int i=0;i<L.H*L.W;i++){
        if(s.boxes.test(i)) boxes.push_back(i);
        if(L.targets.test(i)) targets.push_back(i);
    }
    int n=boxes.size(), m=targets.size();
    if(n==0) return 0;
    if(m==0 || n > m) return 1e9;
    
    int total_cost = 0;
    vector<bool> used_targets(m, false);
    
    // 貪心：每個箱子匹配最近的未使用目標
    for(int b : boxes){
        int br=b/L.W, bc=b%L.W;
        int best_dist = 1e9, best_target = -1;
        
        for(int j=0; j<m; j++){
            if(used_targets[j]) continue;
            int t = targets[j];
            int tr=t/L.W, tc=t%L.W;
            int dist = abs(br-tr) + abs(bc-tc);
            if(dist < best_dist){
                best_dist = dist;
                best_target = j;
            }
        }
        
        if(best_target != -1){
            used_targets[best_target] = true;
            total_cost += best_dist;
        }
    }
    return total_cost;
}

// ===== Heuristic（匈牙利：箱子→目標的最小匹配，成本=曼哈頓） =====
int heuristic(const State&s,const LevelInfo&L){
    // 對於大的狀態，使用簡化版本以提高速度
    if(s.boxes.count() > 15) {
        return simple_heuristic(s, L);
    }
    
    vector<int> boxes, targets;
    for(int i=0;i<L.H*L.W;i++){
        if(s.boxes.test(i)) boxes.push_back(i);
        if(L.targets.test(i)) targets.push_back(i);
    }
    int n=boxes.size(), m=targets.size();
    if(n==0) return 0;
    
    const int INF=1e9;
    if(m==0) return INF; // 有箱子但沒有目標，無解
    
    // 如果箱子數量大於目標數量，這是不可能的狀態
    if(n > m) return INF;
    
    int N=max(n,m);
    // 使用一維數組以改善緩存局部性
    vector<int> a((N+1)*(N+1), INF);
    auto get_a = [&](int i, int j) -> int& { return a[i*(N+1)+j]; };
    
    for(int i=0;i<n;i++){
        int br=boxes[i]/L.W, bc=boxes[i]%L.W;
        for(int j=0;j<m;j++){
            int tr=targets[j]/L.W, tc=targets[j]%L.W;
            get_a(i+1,j+1) = abs(br-tr)+abs(bc-tc);
        }
    }
    
    // 匈牙利算法
    vector<int> u(N+1), v(N+1), p(N+1), way(N+1);
    for(int i=1;i<=n;i++){
        p[0]=i; 
        vector<int> minv(N+1,INF); 
        vector<bool> used(N+1,false);
        int j0=0;
        do{
            used[j0]=true; 
            int i0=p[j0],delta=INF,j1=0;
            for(int j=1;j<=m;j++) if(!used[j]){
                int cur=get_a(i0,j)-u[i0]-v[j];
                if(cur<minv[j]){minv[j]=cur; way[j]=j0;}
                if(minv[j]<delta){delta=minv[j]; j1=j;}
            }
            for(int j=0;j<=m;j++)
                if(used[j]){u[p[j]]+=delta; v[j]-=delta;}
                else minv[j]-=delta;
            j0=j1;
        }while(p[j0]!=0);
        do{int j1=way[j0]; p[j0]=p[j1]; j0=j1;}while(j0);
    }
    return -v[0];
}

void build_tunnel_info(LevelInfo& L){
    for(auto& b : L.tunnel_forward) b.reset();
    auto corridor_cell = [&](int pos, int dir)->bool{
        if(!cell_allows_box(L, pos)) return false;
        int r = pos / L.W;
        int c = pos % L.W;
        auto blocked = [&](int dir_idx){
            int nr = r + DR[dir_idx];
            int nc = c + DC[dir_idx];
            if(!inb(nr,nc,L.H,L.W)) return true;
            int id = idx(nr,nc,L.W);
            return !cell_allows_box(L, id);
        };
        int left_dir = (dir + 1) % 4;
        int right_dir = (dir + 3) % 4;
        if(!blocked(left_dir)) return false;
        if(!blocked(right_dir)) return false;
        return true;
    };

    for(int pos=0; pos<L.H*L.W; ++pos){
        if(!cell_allows_box(L, pos)) continue;
        for(int d=0; d<4; ++d){
            int r = pos / L.W;
            int c = pos % L.W;
            int nr = r + DR[d];
            int nc = c + DC[d];
            if(!inb(nr,nc,L.H,L.W)) continue;
            int next = idx(nr,nc,L.W);
            if(!cell_allows_box(L, next)) continue;
            if(corridor_cell(pos, d) && corridor_cell(next, d)){
                L.tunnel_forward[d].set(pos);
            }
        }
    }
}

// ===== static dead + corridor =====
void build_static_dead(LevelInfo& L){
    const int H=L.H,W=L.W,N=H*W;
    auto inb2=[&](int r,int c){return 0<=r&&r<H&&0<=c&&c<W;};
    auto id=[&](int r,int c){return r*W+c;};
    bitset<MAXC> box_ok, man_ok, is_target;  // 改用 bitset
    for(int i=0;i<N;i++){
        bool wall=L.is_wall.test(i); bool fragile=L.is_fragile.test(i);  // 改用 test()
        if(!wall) man_ok.set(i);                                          // bitset 用 set()
        if(!wall && !fragile) box_ok.set(i);                              // bitset 用 set()
        if(L.targets.test(i)) is_target.set(i);                           // bitset 用 set()
    }
    // reverse-pull
    bitset<MAXC> good; queue<int> q;  // 改用 bitset
    for(int i=0;i<N;i++) if(is_target.test(i)&&box_ok.test(i)){good.set(i); q.push(i);}
    while(!q.empty()){
        int cur=q.front(); q.pop(); int r=cur/W,c=cur%W;
        for(int d=0;d<4;d++){
            int br=r+DR[d], bc=c+DC[d];
            int pr=br+DR[d], pc=bc+DC[d];
            if(!inb2(br,bc)||!inb2(pr,pc)) continue;
            int b=id(br,bc), p=id(pr,pc);
            if(!box_ok.test(b)||!man_ok.test(p)) continue;  // 改用 test()
            if(!good.test(b)){good.set(b); q.push(b);}       // 改用 test() 和 set()
        }
    }
    L.static_dead.reset();  // bitset 用 reset() 清零
    for(int i=0;i<N;i++) if(box_ok.test(i) && !good.test(i)) L.static_dead.set(i);  // 改用 test() 和 set()

    // one-tile corridors (sealed both ends, no targets)
    L.corridor_dead.reset();  // bitset 用 reset() 清零
    auto is_corr_h=[&](int r,int c){
        if(!inb2(r,c)) return false; 
        int i=id(r,c);
        if(L.is_wall.test(i)||L.targets.test(i)||!box_ok.test(i)) return false;
        if(!inb2(r-1,c)||!inb2(r+1,c)) return false;
        return L.is_wall.test(id(r-1,c)) && L.is_wall.test(id(r+1,c));
    };
    auto is_corr_v=[&](int r,int c){
        if(!inb2(r,c)) return false; 
        int i=id(r,c);
        if(L.is_wall.test(i)||L.targets.test(i)||!box_ok.test(i)) return false;
        if(!inb2(r,c-1)||!inb2(r,c+1)) return false;
        return L.is_wall.test(id(r,c-1)) && L.is_wall.test(id(r,c+1));
    };
    for(int r=0;r<H;r++){
        int c=0; while(c<W){
            if(!is_corr_h(r,c)){c++; continue;}
            int Lc=c; while(c<W && is_corr_h(r,c)) c++; int Rc=c-1;
            bool capL = (Lc-1<0) ? true : L.is_wall[id(r,Lc-1)];
            bool capR = (Rc+1>=W) ? true : L.is_wall[id(r,Rc+1)];
            if(capL && capR) for(int k=Lc;k<=Rc;k++) L.corridor_dead[id(r,k)]=1;
        }
    }
    for(int c=0;c<W;c++){
        int r=0; while(r<H){
            if(!is_corr_v(r,c)){r++; continue;}
            int Ur=r; while(r<H && is_corr_v(r,c)) r++; int Dr=r-1;
            bool capU = (Ur-1<0) ? true : L.is_wall[id(Ur-1,c)];
            bool capD = (Dr+1>=H) ? true : L.is_wall[id(Dr+1,c)];
            if(capU && capD) for(int k=Ur;k<=Dr;k++) L.corridor_dead[id(k,c)]=1;
        }
    }
}
bool is_corner_cell(int id,const LevelInfo&L){
    if(L.targets[id])return false;
    int r=id/L.W,c=id%L.W;
    bool up=(r==0)||L.is_wall.test(idx(r-1,c,L.W));
    bool dn=(r==L.H-1)||L.is_wall.test(idx(r+1,c,L.W));
    bool lf=(c==0)||L.is_wall.test(idx(r,c-1,L.W));
    bool rt=(c==L.W-1)||L.is_wall.test(idx(r,c+1,L.W));
    return (up&&lf)||(up&&rt)||(dn&&lf)||(dn&&rt);
}
void build_groove_structures(LevelInfo& L){
    L.grooves.clear();
    
    // 檢測水平凹槽（上下被牆包圍的橫向空間）
    for(int r=1; r<L.H-1; r++){
        for(int c=0; c<L.W; c++){
            if(L.is_wall[idx(r,c,L.W)]) continue;
            
            // 找連續的水平空間
            int start_c = c;
            vector<int> horizontal_space;
            
            while(c < L.W && !L.is_wall[idx(r,c,L.W)]){
                horizontal_space.push_back(idx(r,c,L.W));
                c++;
            }
            
            if(horizontal_space.size() >= 2 && horizontal_space.size() <= 4){
                // 檢查這個水平空間是否被上下包圍形成凹槽
                bool top_wall = true, bottom_wall = true;
                
                for(int pos : horizontal_space){
                    int pr = pos/L.W, pc = pos%L.W;
                    if(!L.is_wall[idx(pr-1,pc,L.W)]) top_wall = false;
                    if(!L.is_wall[idx(pr+1,pc,L.W)]) bottom_wall = false;
                }
                
                // 檢查左右是否有牆或邊界
                bool left_blocked = (start_c == 0) || L.is_wall[idx(r,start_c-1,L.W)];
                bool right_blocked = (c == L.W) || L.is_wall[idx(r,c,L.W)];
                
                // 排除 corner 情況：如果只有單一位置且四周都被牆包圍
                bool is_corner_case = false;
                if(horizontal_space.size() == 1){
                    int pos = horizontal_space[0];
                    if(is_corner_cell(pos, L)){
                        is_corner_case = true;
                    }
                }
                
                if((top_wall || bottom_wall) && left_blocked && right_blocked && !is_corner_case){
                    LevelInfo::Groove groove;
                    groove.positions = horizontal_space;
                    
                    // 入口是左右的位置（如果不是牆的話）
                    if(!left_blocked) groove.entrances.push_back(idx(r,start_c-1,L.W));
                    if(!right_blocked) groove.entrances.push_back(idx(r,c,L.W));
                    
                    L.grooves.push_back(groove);
                }
            }
            c--; // 調整因為外層迴圈會++
        }
    }
    
    // 檢測垂直凹槽（左右被牆包圍的縱向空間）
    for(int c=1; c<L.W-1; c++){
        for(int r=0; r<L.H; r++){
            if(L.is_wall[idx(r,c,L.W)]) continue;
            
            int start_r = r;
            vector<int> vertical_space;
            
            while(r < L.H && !L.is_wall[idx(r,c,L.W)]){
                vertical_space.push_back(idx(r,c,L.W));
                r++;
            }
            
            if(vertical_space.size() >= 2 && vertical_space.size() <= 4){
                bool left_wall = true, right_wall = true;
                
                for(int pos : vertical_space){
                    int pr = pos/L.W, pc = pos%L.W;
                    if(!L.is_wall[idx(pr,pc-1,L.W)]) left_wall = false;
                    if(!L.is_wall[idx(pr,pc+1,L.W)]) right_wall = false;
                }
                
                bool top_blocked = (start_r == 0) || L.is_wall[idx(start_r-1,c,L.W)];
                bool bottom_blocked = (r == L.H) || L.is_wall[idx(r,c,L.W)];
                
                // 排除 corner 情況
                bool is_corner_case = false;
                if(vertical_space.size() == 1){
                    int pos = vertical_space[0];
                    if(is_corner_cell(pos, L)){
                        is_corner_case = true;
                    }
                }
                
                if((left_wall || right_wall) && top_blocked && bottom_blocked && !is_corner_case){
                    LevelInfo::Groove groove;
                    groove.positions = vertical_space;
                    
                    if(!top_blocked) groove.entrances.push_back(idx(start_r-1,c,L.W));
                    if(!bottom_blocked) groove.entrances.push_back(idx(r,c,L.W));
                    
                    L.grooves.push_back(groove);
                }
            }
            r--; // 調整因為外層迴圈會++
        }
    }
    
    // 不印出凹槽結構的debug訊息
    // cerr << "發現 " << L.grooves.size() << " 個凹槽結構:" << endl;
}


// ===== Deadlocks =====
bool deadlock_corner(const State&s,const LevelInfo&L){
    for(int i=0;i<L.H*L.W;i++)
        if(s.boxes.test(i)&&!L.targets[i]&&is_corner_cell(i,L)){
            return true;
        }
    return false;
}
bool deadlock_freeze2x2(const State&s,const LevelInfo&L){
    for(int r=0;r<L.H-1;r++)for(int c=0;c<L.W-1;c++){
        int a=idx(r,c,L.W),b=idx(r,c+1,L.W),d=idx(r+1,c,L.W),e=idx(r+1,c+1,L.W);
        if(s.boxes.test(a)&&s.boxes.test(b)&&s.boxes.test(d)&&s.boxes.test(e))
            if(!L.targets[a]&&!L.targets[b]&&!L.targets[d]&&!L.targets[e]) {
                return true;
            }
    }
    return false;
}
bool deadlock_static(const State&s,const LevelInfo&L){
    for(int i=0;i<L.H*L.W;i++) if(s.boxes.test(i)&&L.static_dead.test(i)) {  // 改用 test()
        return true;
    }
    return false;
}
bool deadlock_corridor(const State&s,const LevelInfo&L){
    for(int i=0;i<L.H*L.W;i++) if(s.boxes.test(i)&&L.corridor_dead.test(i)) {  // 改用 test()
        return true;
    }
    return false;
}
bool isBlockedOnAxis(int boxIdx, const LevelInfo& L, const bitset<MAXC>& boxes,
                     bool vertical, unordered_set<int>& visited) {
    if (visited.count(boxIdx)) return false; // 避免循環 → 視為牆
    visited.insert(boxIdx);

    int r = boxIdx / L.W;
    int c = boxIdx % L.W;

    vector<int> neighbors;
    if (vertical) {
        // 上下
        if (r > 0) neighbors.push_back((r - 1) * L.W + c);
        if (r + 1 < L.H) neighbors.push_back((r + 1) * L.W + c);
    } else {
        // 左右
        if (c > 0) neighbors.push_back(r * L.W + (c - 1));
        if (c + 1 < L.W) neighbors.push_back(r * L.W + (c + 1));
    }

    // case 1: 鄰居有牆
    bool wallBlock = false;
    for (int n : neighbors) {
        if (L.is_wall.test(n)) wallBlock = true;
        if(visited.count(n)) wallBlock = true; // 視為牆
    }
    if (wallBlock) return true;

    // case 2: 兩側都是 simple dead
    if (neighbors.size() == 2) {
        if (L.static_dead.test(neighbors[0]) && boxes.test(neighbors[0])&&boxes.test(neighbors[1])&& L.static_dead.test(neighbors[1])) {
            return true;
        }
    }

    // case 3: 遞迴檢查鄰居箱子
    for (int n : neighbors) {
        if (boxes.test(n)) {
            if (isBlockedOnAxis(n, L, boxes, !vertical, visited)) {
                return true;
            }
        }
    }

    return false; // 沒被擋
}

// 判斷某個箱子是否凍結
bool isFrozenBox(int idx, const LevelInfo& L, const bitset<MAXC>& boxes) {
    if(L.targets.test(idx)) return false; // 目標上的箱子不算凍結
    unordered_set<int> visited;
    bool verticalBlocked = isBlockedOnAxis(idx, L, boxes, true, visited);
    visited.clear();
    bool horizontalBlocked = isBlockedOnAxis(idx, L, boxes, false, visited);
    return verticalBlocked && horizontalBlocked;
}
bool deadlock_freeze(const State&s,const LevelInfo&L){
    for(int i=0;i<L.H*L.W;i++) if(s.boxes.test(i)&&isFrozenBox(i,L,s.boxes)) {  // 改用 test()
        return true;
    }
    return false;
}

// 偵測箱子堵住凹槽入口導致目標無法到達的 deadlock
bool deadlock_blocked_groove_access(const State&s,const LevelInfo&L){
    // 檢查是否有目標位置被箱子群組堵住無法到達
    for(int target_pos=0; target_pos<L.H*L.W; target_pos++){
        if(!L.targets[target_pos] || s.boxes.test(target_pos)) continue; // 跳過非目標或已有箱子的位置
        
        int tr = target_pos/L.W, tc = target_pos%L.W;
        
        // 檢查這個目標位置是否在一個凹槽中（被牆圍住）
        bool in_groove = false;
        int groove_entrance_r = -1, groove_entrance_c = -1;
        
        // 檢查目標是否在水平凹槽中（上下有牆，某一側有開口）
        bool top_wall = (tr == 0) || L.is_wall[idx(tr-1,tc,L.W)];
        bool bottom_wall = (tr == L.H-1) || L.is_wall[idx(tr+1,tc,L.W)];
        
        if(top_wall || bottom_wall){
            // 檢查左右哪邊是入口
            bool left_blocked = (tc == 0) || L.is_wall[idx(tr,tc-1,L.W)];
            bool right_blocked = (tc == L.W-1) || L.is_wall[idx(tr,tc+1,L.W)];
            
            if(left_blocked && !right_blocked){
                in_groove = true;
                groove_entrance_r = tr;
                groove_entrance_c = tc + 1;
            }
            else if(right_blocked && !left_blocked){
                in_groove = true;
                groove_entrance_r = tr;
                groove_entrance_c = tc - 1;
            }
        }
        
        // 檢查目標是否在垂直凹槽中（左右有牆，某一側有開口）
        bool left_wall = (tc == 0) || L.is_wall[idx(tr,tc-1,L.W)];
        bool right_wall = (tc == L.W-1) || L.is_wall[idx(tr,tc+1,L.W)];
        
        if(left_wall || right_wall){
            bool top_blocked = (tr == 0) || L.is_wall[idx(tr-1,tc,L.W)];
            bool bottom_blocked = (tr == L.H-1) || L.is_wall[idx(tr+1,tc,L.W)];
            
            if(top_blocked && !bottom_blocked){
                in_groove = true;
                groove_entrance_r = tr + 1;
                groove_entrance_c = tc;
            }
            else if(bottom_blocked && !top_blocked){
                in_groove = true;
                groove_entrance_r = tr - 1;
                groove_entrance_c = tc;
            }
        }
        
        if(in_groove && inb(groove_entrance_r,groove_entrance_c,L.H,L.W)){
            // 檢查凹槽入口附近是否有箱子堵住
            vector<pair<int,int>> blocking_boxes;
            
            // 檢查入口位置及其周圍的箱子
            for(int dr=-1; dr<=1; dr++){
                for(int dc=-1; dc<=1; dc++){
                    int check_r = groove_entrance_r + dr;
                    int check_c = groove_entrance_c + dc;
                    if(!inb(check_r,check_c,L.H,L.W)) continue;
                    
                    int check_pos = idx(check_r,check_c,L.W);
                    if(s.boxes.test(check_pos) && !L.targets[check_pos]){
                        blocking_boxes.push_back({check_r, check_c});
                    }
                }
            }
            
            if(blocking_boxes.size() >= 2){
                // 檢查這些箱子是否形成了一個無法移動的障礙
                bool all_immobile = true;
                for(auto [br, bc] : blocking_boxes){
                    bool can_move_this_box = false;
                    
                    for(int d=0; d<4; d++){
                        int push_from_r = br - DR[d], push_from_c = bc - DC[d];
                        int push_to_r = br + DR[d], push_to_c = bc + DC[d];
                        
                        if(!inb(push_from_r,push_from_c,L.H,L.W) || !inb(push_to_r,push_to_c,L.H,L.W)) continue;
                        
                        int push_from_pos = idx(push_from_r,push_from_c,L.W);
                        int push_to_pos = idx(push_to_r,push_to_c,L.W);
                        
                        if(!L.is_wall[push_to_pos] && !L.is_fragile[push_to_pos] && 
                           !s.boxes.test(push_to_pos) && !L.is_wall[push_from_pos] && 
                           !s.boxes.test(push_from_pos)){
                            can_move_this_box = true;
                            break;
                        }
                    }
                    
                    if(can_move_this_box){
                        all_immobile = false;
                        break;
                    }
                }
                
                if(all_immobile){
                    // 不印出block groove access的debug訊息
                    // cerr << "DEADLOCK: Blocked groove access - target at (" << tr << "," << tc 
                    //      << ") is unreachable due to immobile boxes blocking entrance:";
                    // for(auto [br, bc] : blocking_boxes){
                    //     cerr << " (" << br << "," << bc << ")";
                    // }
                    // cerr << endl;
                    // print_state(s, L, "Blocked Groove Access Deadlock Detected");
                    return true;
                }
            }
        }
    }
    return false;
}

// 偵測凹槽入口被堵住的 deadlock  
bool deadlock_filled_grooves(const State&s,const LevelInfo&L){
    for(size_t i=0; i<L.grooves.size(); i++){
        const auto& groove = L.grooves[i];
        
        // 檢查凹槽內是否有任何目標位置
        bool has_targets_in_groove = false;
        for(int groove_pos : groove.positions){
            if(L.targets[groove_pos]){
                has_targets_in_groove = true;
                break;
            }
        }
        
        // 只有當凹槽內沒有目標時，才檢查是否被外圍箱子堵住
        if(has_targets_in_groove) continue;
        
        // 檢查玩家是否在凹槽內，如果是則不算 deadlock
        bool player_in_groove = false;
        for(int groove_pos : groove.positions){
            if(groove_pos == s.player){
                player_in_groove = true;
                break;
            }
        }
        
        // 如果玩家在凹槽內，則不算 deadlock（玩家可以從內部推動箱子）
        if(player_in_groove) continue;
        
        // 檢查 groove 的直接入口（相鄰位置）是否被完全堵住
        vector<pair<int,int>> entrance_positions;
        vector<pair<int,int>> blocking_boxes;
        
        // 找到 groove 的所有入口位置（相鄰且不是牆的位置）
        for(int groove_pos : groove.positions){
            int gr = groove_pos/L.W, gc = groove_pos%L.W;
            
            for(int d=0; d<4; d++){
                int er = gr + DR[d], ec = gc + DC[d];
                if(!inb(er,ec,L.H,L.W)) continue;
                
                int entrance_pos = idx(er,ec,L.W);
                
                // 如果入口位置不是牆且不在 groove 內
                if(!L.is_wall[entrance_pos]){
                    bool is_in_groove = false;
                    for(int gpos : groove.positions){
                        if(gpos == entrance_pos){
                            is_in_groove = true;
                            break;
                        }
                    }
                    if(!is_in_groove){
                        entrance_positions.push_back({er, ec});
                    }
                }
            }
        }
        
        // 移除重複的入口位置
        sort(entrance_positions.begin(), entrance_positions.end());
        entrance_positions.erase(unique(entrance_positions.begin(), entrance_positions.end()), entrance_positions.end());
        
        // 檢查所有入口是否都被箱子堵住
        bool all_entrances_blocked = true;
        for(auto [er, ec] : entrance_positions){
            int entrance_pos = idx(er,ec,L.W);
            if(s.boxes.test(entrance_pos)){
                blocking_boxes.push_back({er, ec});
            } else {
                all_entrances_blocked = false;
            }
        }
        
        // 如果所有入口都被箱子堵住時才算 deadlock
        if(all_entrances_blocked && !blocking_boxes.empty()){
            return true;
        }
    }
    return false;
}

// 偵測：沿著內層邊界（牆壁的一側為箱子可移動區）分段，若某段上「貼牆的箱子數」大於「該段上的目標數」則為 deadlock
// 實作說明：
// - 內層邊界定義為：某格地面 f（可放箱子：非牆且非fragile），其四鄰有牆，則 f 與該牆構成一個邊界點
// - 依四個方向將「同一面牆」的連續邊界點視為一段（遇到拐彎就結束一段），針對每段計數 boxes 與 targets
// - 若 boxes > targets 則回傳 true
bool deadlock_boundary_wall_segments(const State& s, const LevelInfo& L){
    const int H=L.H, W=L.W;
    auto is_box_ok = [&](int p){ return !L.is_wall[p] && !L.is_fragile[p]; };

    // 垂直方向：牆在左側（掃描每一欄，按列形成直段）
    for(int c=1; c<W; ++c){
        int r=0;
        while(r<H){
            // 一段的起點：floor 可放箱 + 左側是牆
            auto cond = [&](int rr){ int f=idx(rr,c,W); return is_box_ok(f) && L.is_wall[idx(rr,c-1,W)]; };
            if(!cond(r)){ ++r; continue; }
            int r0=r; while(r<H && cond(r)) ++r; int r1=r-1; // [r0,r1] 形成一段
            if(r1>=r0){
                // 新規則：不論封閉與否，只要此段有相鄰兩格皆為箱子（xx 或 xX），但不是兩個都在目標上（XX），判定 deadlock
                for(int rr=r0; rr<r1; ++rr){
                    int f1=idx(rr,c,W), f2=idx(rr+1,c,W);
                    if(s.boxes.test(f1) && s.boxes.test(f2)) {
                        // 如果兩個箱子都在目標上，則不算 deadlock
                        if(!(L.targets.test(f1) && L.targets.test(f2))) {
                            return true;
                        }
                    }
                }
                // 判斷是否為封閉段：兩端若有任何一端是「開口」則忽略（不判 deadlock）
                bool closed_top=true, closed_bottom=true;
                // 段的上一格
                if(r0-1>=0){
                    int f_up = idx(r0-1,c,W);
                    if(is_box_ok(f_up)){
                        // cond 在此為 false 才會斷段，代表左側不是牆或 floor 不可放；若是可放且左側不是牆，視為開口
                        if(!L.is_wall[idx(r0-1,c-1,W)]) closed_top=false;
                    }
                }
                // 段的下一格
                if(r1+1<H){
                    int f_dn = idx(r1+1,c,W);
                    if(is_box_ok(f_dn)){
                        if(!L.is_wall[idx(r1+1,c-1,W)]) closed_bottom=false;
                    }
                }
                if(!(closed_top && closed_bottom)){
                    continue; // 非封閉區間：可從端點推出去，忽略
                }
                int boxes=0, targets=0;
                for(int rr=r0; rr<=r1; ++rr){ int f=idx(rr,c,W); if(s.boxes.test(f)) ++boxes; if(L.targets.test(f)) ++targets; }
                if(boxes>targets){  return true; }
            }
        }
    }

    // 垂直方向：牆在右側
    for(int c=0; c<W-1; ++c){
        int r=0;
        while(r<H){
            auto cond = [&](int rr){ int f=idx(rr,c,W); return is_box_ok(f) && L.is_wall[idx(rr,c+1,W)]; };
            if(!cond(r)){ ++r; continue; }
            int r0=r; while(r<H && cond(r)) ++r; int r1=r-1;
            if(r1>=r0){
                for(int rr=r0; rr<r1; ++rr){
                    int f1=idx(rr,c,W), f2=idx(rr+1,c,W);
                    if(s.boxes.test(f1) && s.boxes.test(f2)) {
                        // 如果兩個箱子都在目標上，則不算 deadlock
                        if(!(L.targets.test(f1) && L.targets.test(f2))) {
                            return true;
                        }
                    }
                }
                bool closed_top=true, closed_bottom=true;
                if(r0-1>=0){
                    int f_up = idx(r0-1,c,W);
                    if(is_box_ok(f_up)){
                        if(!L.is_wall[idx(r0-1,c+1,W)]) closed_top=false;
                    }
                }
                if(r1+1<H){
                    int f_dn = idx(r1+1,c,W);
                    if(is_box_ok(f_dn)){
                        if(!L.is_wall[idx(r1+1,c+1,W)]) closed_bottom=false;
                    }
                }
                if(!(closed_top && closed_bottom)){
                    continue;
                }
                int boxes=0, targets=0;
                for(int rr=r0; rr<=r1; ++rr){ int f=idx(rr,c,W); if(s.boxes.test(f)) ++boxes; if(L.targets.test(f)) ++targets; }
                if(boxes>targets){return true; }
            }
        }
    }

    // 水平方向：牆在上側（掃描每一列，按行形成直段）
    for(int r=1; r<H; ++r){
        int c=0;
        while(c<W){
            auto cond = [&](int cc){ int f=idx(r,cc,W); return is_box_ok(f) && L.is_wall[idx(r-1,cc,W)]; };
            if(!cond(c)){ ++c; continue; }
            int c0=c; while(c<W && cond(c)) ++c; int c1=c-1;
            if(c1>=c0){
                for(int cc=c0; cc<c1; ++cc){
                    int f1=idx(r,cc,W), f2=idx(r,cc+1,W);
                    if(s.boxes.test(f1) && s.boxes.test(f2)) {
                        // 如果兩個箱子都在目標上，則不算 deadlock
                        if(!(L.targets.test(f1) && L.targets.test(f2))) {
                            return true;
                        }
                    }
                }
                bool closed_left=true, closed_right=true;
                if(c0-1>=0){
                    int f_l = idx(r,c0-1,W);
                    if(is_box_ok(f_l)){
                        if(!L.is_wall[idx(r-1,c0-1,W)]) closed_left=false;
                    }
                }
                if(c1+1<W){
                    int f_r = idx(r,c1+1,W);
                    if(is_box_ok(f_r)){
                        if(!L.is_wall[idx(r-1,c1+1,W)]) closed_right=false;
                    }
                }
                if(!(closed_left && closed_right)){
                    continue;
                }
                int boxes=0, targets=0;
                for(int cc=c0; cc<=c1; ++cc){ int f=idx(r,cc,W); if(s.boxes.test(f)) ++boxes; if(L.targets.test(f)) ++targets; }
                if(boxes>targets){return true; }
            }
        }
    }

    // 水平方向：牆在下側
    for(int r=0; r<H-1; ++r){
        int c=0;
        while(c<W){
            auto cond = [&](int cc){ int f=idx(r,cc,W); return is_box_ok(f) && L.is_wall[idx(r+1,cc,W)]; };
            if(!cond(c)){ ++c; continue; }
            int c0=c; while(c<W && cond(c)) ++c; int c1=c-1;
            if(c1>=c0){
                for(int cc=c0; cc<c1; ++cc){
                    int f1=idx(r,cc,W), f2=idx(r,cc+1,W);
                    if(s.boxes.test(f1) && s.boxes.test(f2)) {
                        // 如果兩個箱子都在目標上，則不算 deadlock
                        if(!(L.targets.test(f1) && L.targets.test(f2))) {
                            return true;
                        }
                    }
                }
                bool closed_left=true, closed_right=true;
                if(c0-1>=0){
                    int f_l = idx(r,c0-1,W);
                    if(is_box_ok(f_l)){
                        if(!L.is_wall[idx(r+1,c0-1,W)]) closed_left=false;
                    }
                }
                if(c1+1<W){
                    int f_r = idx(r,c1+1,W);
                    if(is_box_ok(f_r)){
                        if(!L.is_wall[idx(r+1,c1+1,W)]) closed_right=false;
                    }
                }
                if(!(closed_left && closed_right)){
                    continue;
                }
                int boxes=0, targets=0;
                for(int cc=c0; cc<=c1; ++cc){ int f=idx(r,cc,W); if(s.boxes.test(f)) ++boxes; if(L.targets.test(f)) ++targets; }
                if(boxes>targets){  return true; }
            }
        }
    }

    return false;
}
// 快速 deadlock 檢查：只檢查最基本的情況
inline bool fast_deadlock_check(const State&s,const LevelInfo&L){
    // 只檢查最快速的 deadlock 類型
    if(deadlock_static(s,L)) return true;
    if(deadlock_corridor(s,L)) return true;
    if(deadlock_corner(s,L)) return true;
    if(deadlock_freeze2x2(s,L)) return true;
    return false;
}
inline bool big_deadlock(const State&s,const LevelInfo&L){
    
    // 較慢的檢測放後面
    // if(deadlock_boundary_wall_segments(s,L)) return true;
    if(deadlock_freeze(s,L)) return true;
    if(deadlock_filled_grooves(s,L)) return true;
    if(deadlock_blocked_groove_access(s,L)) return true;
    return false;
}




inline bool any_deadlock(const State&s,const LevelInfo&L){
    if(deadlock_static(s,L)) return true;
    if(deadlock_corridor(s,L)) return true;
    if(deadlock_corner(s,L)) return true;
    if(deadlock_freeze2x2(s,L)) return true;
    // 較慢的檢測放後面
    if(deadlock_boundary_wall_segments(s,L)) return true;
    if(deadlock_filled_grooves(s,L)) return true;
    if(deadlock_blocked_groove_access(s,L)) return true;
    return false;
}

// ===== MergeState（作為 key） =====
struct MergeState {
    bitset<MAXC> boxes;          // 所有箱子的位置
    int playerPos;               // 玩家位置（用於路徑重建和 BFS 起點）
};

// 簡化的 hash，只使用箱子位置
struct BoxOnlyState {
    bitset<MAXC> boxes;
};

struct BoxOnlyHash {
    size_t operator()(const BoxOnlyState& s) const noexcept {
        size_t h=0;
        // hash boxes (只hash箱子位置)
        for(size_t block=0; block<(MAXC+63)/64; ++block){
            uint64_t v=0;
            for(size_t b=0; b<64 && block*64+b<MAXC; ++b)
                if(s.boxes.test(block*64+b)) v|=(1ull<<b);
            boost::hash_combine(h, v);
        }
        return h;
    }
};

struct BoxOnlyEq {
    bool operator()(const BoxOnlyState&a,const BoxOnlyState&b) const {
        return a.boxes==b.boxes;
    }
};

// 存儲信息：對於每個箱子配置，保存所有不連通的玩家位置及其最佳代價
struct PlayerStateInfo {
    int g_value;
    int player_pos;
    // 移除 full_state，因為它重複了 BoxOnlyState 中的 boxes 信息，可以動態重建
};

// 線程安全的狀態存儲結構
struct ThreadSafePlayerStates {
    vector<PlayerStateInfo> states;
    std::mutex mutex;  // 使用 std::mutex 來保護 vector 操作
};

struct MergeHash {
    size_t operator()(const MergeState& s) const noexcept {
        size_t h=0;
        // 只 hash 箱子位置，不包含玩家位置
        for(int i=0;i<MAXC;i++) if(s.boxes.test(i))
            h ^= std::hash<int>()(i) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};
struct MergeEq {
    bool operator()(const MergeState&a,const MergeState&b) const {
        return a.boxes==b.boxes && a.playerPos==b.playerPos;
    }
};

// 檢查兩個玩家位置在當前箱子配置下是否連通
bool are_players_connected(int pos1, int pos2, const bitset<MAXC>& boxes, const LevelInfo& L) {
    if (pos1 == pos2) return true;
    
    bitset<MAXC> vis;  // 改用 bitset
    queue<int> q;
    q.push(pos1);
    vis.set(pos1);  // bitset 用 set()
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        if (u == pos2) return true;
        
        int ur = u / L.W, uc = u % L.W;
        for (int d = 0; d < 4; d++) {
            int vr = ur + DR[d], vc = uc + DC[d];
            if (!inb(vr, vc, L.H, L.W)) continue;
            int v = idx(vr, vc, L.W);
            if (vis.test(v)) continue;         // bitset 用 test()
            if (L.is_wall.test(v)) continue;   // bitset 用 test()
            if (boxes.test(v)) continue;  // 不能穿過箱子
            vis.set(v);                   // bitset 用 set()
            q.push(v);
        }
    }
    return false;
}

// ===== parent map：記住每一步的「走路＋推」 =====
struct ParentInfo {
    MergeState parent;
    string move; // 從 parent → child 的完整動作
};
// 線程安全的 parent_map
tbb::concurrent_unordered_map<MergeState, ParentInfo, MergeHash, MergeEq> parent_map;

// reconstruct：一路把 move 串起來
string reconstruct_full(MergeState goal, MergeState start){
    string path;
    auto cur=goal;
    while(!(cur.playerPos==start.playerPos && cur.boxes==start.boxes)){
        auto it=parent_map.find(cur);
        if(it==parent_map.end()) break; // safety
        path = it->second.move + path;
        cur = it->second.parent;
    }
    return path;
}

// ===== A* =====
struct Node{
    MergeState s;
    int g,h;
    bool operator<(const Node&o) const { return g+h > o.g+o.h; }
};

string solve_astar(LevelInfo& L){
    // 使用6個線程（用戶指定）
    unsigned int num_threads = 6;
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
    
    build_static_dead(L);
    build_groove_structures(L);
    build_tunnel_info(L);

    // 為6線程預分配更大的parent_map容量
    parent_map.reserve(500000);

    MergeState start{L.boxes, L.player};
    if(PRINT_ALL_STATES) {
        print_state({start.playerPos,start.boxes}, L, "Start State");
    }

    if(L.boxes==L.targets) return "";

    // 使用箱子配置作為主鍵，存儲多個玩家狀態
    tbb::concurrent_unordered_map<BoxOnlyState, ThreadSafePlayerStates, BoxOnlyHash, BoxOnlyEq> box_states;

    // 初始狀態
    BoxOnlyState start_box{L.boxes};
    {
        ThreadSafePlayerStates& tsps = box_states[start_box];
        std::lock_guard<std::mutex> lock(tsps.mutex);
        tsps.states.push_back({0, L.player});
    }

    const size_t BEAM_WIDTH=35000;  // 針對6線程增加 beam width
    const double G_WEIGHT=0;       // 調整權重以更重視路徑長度
    auto scoreVal=[&](int g,int h){return G_WEIGHT*g+h;};

    vector<Node> frontier;
    frontier.push_back({start,0,heuristic({start.playerPos,start.boxes},L)});

    int layer=0;
    while(!frontier.empty()){ 
        ++layer;
        
        for(const auto&nd:frontier){
            if(nd.s.boxes==L.targets){
                return reconstruct_full(nd.s,start);
            }
        }
        
        // 只從最好的 BEAM_WIDTH 個狀態擴展，但保留所有已訪問的狀態
        vector<Node> work=frontier;
        if(work.size()>BEAM_WIDTH){
            nth_element(work.begin(),work.begin()+BEAM_WIDTH,work.end(),[&](auto&a,auto&b){
                return scoreVal(a.g,a.h)<scoreVal(b.g,b.h);
            });
            work.resize(BEAM_WIDTH);  // 只用最好的 BEAM_WIDTH 個來擴展
        }

        struct Succ{MergeState nxt;int ng;int nh;ParentInfo p;};
        tbb::concurrent_vector<Succ> all_succ;
        all_succ.reserve(work.size() * 24); // 針對6線程優化預分配：6*4=24
        int N=L.H*L.W;
        
        // 針對6線程優化grainsize計算
        size_t grainsize = max(1UL, work.size() / 24); // 6線程 * 4批次 = 24
        if(grainsize > 50) grainsize = 50; // 適合6線程的grainsize上限
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0,work.size(),grainsize),
        [&](auto&range){
            for(size_t ii=range.begin();ii<range.end();++ii){
            const Node&cur=work[ii];
            // 玩家可達 BFS (使用 bitset 優化)
            bitset<MAXC> vis;vector<int> pre(N,-1),how(N,-1);  // 改用 bitset
            queue<int> q;q.push(cur.s.playerPos);vis.set(cur.s.playerPos);pre[cur.s.playerPos]=cur.s.playerPos;
            while(!q.empty()){
                int u=q.front();q.pop();
                int ur=u/L.W,uc=u%L.W;
                for(int d=0;d<4;d++){
                    int vr=ur+DR[d],vc=uc+DC[d];
                    if(!inb(vr,vc,L.H,L.W)) continue;
                    int v=idx(vr,vc,L.W);
                    if(vis.test(v)||L.is_wall.test(v)||cur.s.boxes.test(v)) continue;  // 改用 test()
                    vis.set(v);pre[v]=u;how[v]=d;q.push(v);  // 改用 set()
                }
            }
            // 嘗試推動箱子
            for(int b=0;b<N;b++) if(cur.s.boxes.test(b)){
                int br=b/L.W,bc=b%L.W;
                for(int d=0;d<4;d++){
                    int pr=br-DR[d],pc=bc-DC[d];
                    int nr=br+DR[d],nc=bc+DC[d];
                    if(!inb(pr,pc,L.H,L.W)||!inb(nr,nc,L.H,L.W)) continue;
                    int pid=idx(pr,pc,L.W),nid=idx(nr,nc,L.W);
                    if(L.is_wall.test(nid)||L.is_fragile.test(nid)||cur.s.boxes.test(nid)) continue;  // 改用 test()
                    if(L.is_wall.test(pid)||cur.s.boxes.test(pid)) continue;  // 改用 test()
                    if(!vis.test(pid)) continue;  // 改用 test()
                    // reconstruct path to pid (玩家走到箱子推入位置)
                    string walk;
                    for(int x=pid;x!=pre[x];x=pre[x]){
                        walk.push_back(MV[how[x]]);
                    }
                    reverse(walk.begin(),walk.end());
                    
                    // 完整的移動序列：玩家移動 + 推箱動作
                    string full_move = walk + MV[d];
                    bitset<MAXC> next_boxes = cur.s.boxes;
                    next_boxes.reset(b);
                    next_boxes.set(nid);
                    int new_player = b;
                    int box_pos = nid;
                    bool invalid = false;

                    auto validate_state = [&](int player_pos){
                        State st{player_pos, next_boxes};
                        if(fast_deadlock_check(st, L)) return false;
                        if(big_deadlock(st, L)) return false;
                        return true;
                    };

                    if(!validate_state(new_player)) continue;

                    if(!L.targets.test(box_pos)){
                        while(L.tunnel_forward[d].test(box_pos)){
                            int br2 = box_pos / L.W;
                            int bc2 = box_pos % L.W;
                            int nr2 = br2 + DR[d];
                            int nc2 = bc2 + DC[d];
                            if(!inb(nr2, nc2, L.H, L.W)) break;
                            int next_pos = idx(nr2, nc2, L.W);
                            if(!cell_allows_box(L, next_pos)) break;
                            if(cur.s.boxes.test(next_pos)) break;
                            if(next_boxes.test(next_pos)) break;
                            next_boxes.reset(box_pos);
                            next_boxes.set(next_pos);
                            new_player = box_pos;
                            box_pos = next_pos;
                            full_move.push_back(MV[d]);
                            if(!validate_state(new_player)){
                                invalid = true;
                                break;
                            }
                            if(L.targets.test(box_pos)) break;
                        }
                    }
                    if(invalid) continue;
                    if(isFrozenBox(box_pos, L, next_boxes)){
                        continue;
                    }

                    MergeState nxt{next_boxes,new_player};
                    int ng=cur.g+(int)full_move.size();
                    int nh=heuristic({new_player,next_boxes},L);
                    all_succ.push_back({nxt,ng,nh,{cur.s,full_move}});
                }
            }
            }
        });

        tbb::concurrent_vector<Node> candidates;
        candidates.reserve(all_succ.size());
        
        // 性能優化：當狀態數量很大時，跳過一些連通性檢查
        bool skip_connectivity_for_performance = (box_states.size() > 50000);
        
        // 平行化處理所有候選狀態
        size_t process_grainsize = max(1UL, all_succ.size() / 24); // 針對6線程優化
        if(process_grainsize > 100) process_grainsize = 100;
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, all_succ.size(), process_grainsize),
        [&](auto& range) {
            for(size_t i = range.begin(); i < range.end(); ++i) {
                auto& succ = all_succ[i];
                
                // 檢查新狀態
                BoxOnlyState nxt_box{succ.nxt.boxes};
                auto nxt_it = box_states.find(nxt_box);
                
                bool should_add = false;
                if (nxt_it == box_states.end()) {
                    // 全新的箱子配置
                    should_add = true;
                } else {
                    // 已存在相同箱子配置，檢查玩家連通性
                    bool connected_to_existing = false;
                    int best_g = INT_MAX;
                    
                    // 性能優化：當狀態數量很大且這個箱子配置已經有很多玩家位置時，跳過連通性檢查
                    {
                        std::lock_guard<std::mutex> lock(nxt_it->second.mutex);
                        if (skip_connectivity_for_performance && nxt_it->second.states.size() > 5) {
                            // 簡單比較：如果找到完全相同的玩家位置
                            for (const auto& psi : nxt_it->second.states) {
                                if (psi.player_pos == succ.nxt.playerPos) {
                                    connected_to_existing = true;
                                    best_g = min(best_g, psi.g_value);
                                    break;
                                }
                            }
                        } else {
                            // 完整的連通性檢查
                            for (const auto& psi : nxt_it->second.states) {
                                if (are_players_connected(succ.nxt.playerPos, psi.player_pos, succ.nxt.boxes, L)) {
                                    connected_to_existing = true;
                                    best_g = min(best_g, psi.g_value);
                                }
                            }
                        }
                    }
                    
                    if (!connected_to_existing) {
                        // 與現有玩家位置都不連通，添加新狀態
                        should_add = true;
                    } else if (succ.ng < best_g) {
                        // 連通但找到更好路徑，更新
                        should_add = true;
                        // 移除被支配的狀態
                        {
                            std::lock_guard<std::mutex> lock(nxt_it->second.mutex);
                            auto& states = nxt_it->second.states;
                            states.erase(remove_if(states.begin(), states.end(),
                                [&](const PlayerStateInfo& psi) {
                                    return are_players_connected(succ.nxt.playerPos, psi.player_pos, succ.nxt.boxes, L) && succ.ng <= psi.g_value;
                                }), states.end());
                        }
                    } else {
                        // 連通且沒有更好的路徑，跳過
                    }
                }
                
                if (should_add) {
                    {
                        ThreadSafePlayerStates& tsps = box_states[nxt_box];
                        std::lock_guard<std::mutex> lock(tsps.mutex);
                        tsps.states.push_back({succ.ng, succ.nxt.playerPos});
                    }
                    parent_map[succ.nxt] = succ.p;
                    candidates.push_back({succ.nxt, succ.ng, succ.nh});
                }
            }
        });
        
        // 將 concurrent_candidates 轉換為一般 vector
        vector<Node> regular_candidates(candidates.begin(), candidates.end());
        
        // 排序
        sort(regular_candidates.begin(),regular_candidates.end(),[&](auto&a,auto&b){
            return scoreVal(a.g,a.h)<scoreVal(b.g,b.h);
        });
        
        if(regular_candidates.empty()) return "";
        
        // 只保留最好的 BEAM_WIDTH 個狀態
        if(regular_candidates.size() > BEAM_WIDTH) {
            regular_candidates.resize(BEAM_WIDTH);
        }
        
        // 將候選狀態放入下一輪的 frontier
        frontier.swap(regular_candidates);
    }
    return "";
}

int main(int argc,char**argv){
    ios::sync_with_stdio(false);cin.tie(nullptr);
    if(argc<2){cout<<"\n";return 0;}
    try{
        LevelInfo L=load_level(argv[1]);
        string ans=solve_astar(L);
        cout<<ans<<"\n";
    }catch(const Oops& e){
        cerr<<e.what()<<"\n"; cout<<"\n";
    }
    return 0;
}
