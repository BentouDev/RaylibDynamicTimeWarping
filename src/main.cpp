#include "raylib.h"
#include "raymath.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include <vector>
#include <span>
#include <memory>
#include <format>
#include <algorithm>
#include <assert.h>

// --
// MEMORY

template <typename T>
T* align_ptr(std::size_t alignment, std::size_t size, void* ptr, std::size_t space)
{
    return reinterpret_cast<T*>(std::align(alignment, size, ptr, space));
}

#define STACK_ALLOC(type, count) \
    std::span< type > { \
        align_ptr<type> \
        ( \
            alignof(type), \
            sizeof(type) * count, \
            _malloca( alignof(type) + (sizeof(type) * count) ), \
            alignof(type) + (sizeof(type) * count) \
        ), count }

template <typename T>
std::vector<T> CopySpan(std::span<T> spn)
{
    std::vector<T> result;
    result.insert(result.begin(), spn.begin(), spn.end());

    return result;
}

// --
// KERNEL

using uint = unsigned int;

template <typename T>
int IndexOf(std::span<T> array, const T& val)
{
    for (uint i = 0; i < array.size(); i++)
    {
        if (array[i] == val)
        {
            return i;
        }
    }

    return -1;
}

template <typename T, typename Iter>
void RemoveIndicesFromVector(std::vector<T>& v, Iter begin, Iter end)
// requires std::is_convertible_v<std::iterator_traits<Iter>::value_type, std::size_t>
{
    assert(std::is_sorted(begin, end));
    auto rm_iter = begin;
    std::size_t current_index = 0;

    const auto pred = [&](const T&) {
        // any more to remove?
        if (rm_iter == end) { return false; }
        // is this one specified?
        if (*rm_iter == current_index++) { return ++rm_iter, true; }
        return false;
    };

    v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());
}

template <typename T>
// requires std::is_convertible_v<S::value_type, std::size_t>
void RemoveIndicesFromVector(std::vector<T>& v, std::span<uint> rm)
{
    std::span<uint> rm_copy = STACK_ALLOC(uint, rm.size());
    std::copy(rm.begin(), rm.end(), rm_copy.begin());
    std::sort(rm_copy.begin(), rm_copy.end());

    return RemoveIndicesFromVector(v, rm_copy.begin(), rm_copy.end());
}

// --
// TYPES - Dynamic Time Warping

template <typename T>
struct DataMatrixView
{
    DataMatrixView() = default;
    virtual ~DataMatrixView() = default;

    DataMatrixView(T* mem, uint x_dim, uint y_dim)
        : data(mem), x_size(x_dim), y_size(y_dim)
    { }

    DataMatrixView(std::span<T> mem, uint x_dim, uint y_dim)
        : data(mem.data()), x_size(x_dim), y_size(y_dim)
    {
        assert(mem.size() == x_dim * y_dim);
    }

    void fill(const T& val)
    {
        std::fill(data, data + x_size * y_size, val);
    }

    std::span<T> operator[](uint y)
    {
        assert(y < y_size);

        return std::span<T>{ data + (y * x_size), x_size };
    }

    T* data = nullptr;
    uint x_size = 0U;
    uint y_size = 0U;
};

template <typename T>
struct DataMatrixHeap : public DataMatrixView<T>
{
    DataMatrixHeap() = default;

    explicit DataMatrixHeap(const DataMatrixView<T>& view)
        : DataMatrixHeap(view.data, view.x_size, view.y_size)
    {

    }

    // Copy ctor
    DataMatrixHeap(const DataMatrixHeap<T>& other)
        : DataMatrixHeap(other.data, other.x_size, other.y_size)
    {

    }

    // Move ctor
    DataMatrixHeap(DataMatrixHeap<T>&& other)
    {
        this->data = other.data;
        this->x_size = other.x_size;
        this->y_size = other.y_size;

        other.data = nullptr;
        other.x_size = 0U;
        other.y_size = 0U;
    }

    // Copy assign
    DataMatrixHeap<T>& operator=(const DataMatrixHeap& other)
    {
        this->~DataMatrixHeap();
        new (this) DataMatrixHeap(other.data, other.x_size, other.y_size);
        return *this;
    }

    // Move assign
    DataMatrixHeap<T>& operator=(DataMatrixHeap&& other)
    {
        if (&other != this)
        {
            this->~DataMatrixHeap();

            this->data = other.data;
            this->x_size = other.x_size;
            this->y_size = other.y_size;

            other.data = nullptr;
            other.x_size = 0U;
            other.y_size = 0U;
        }

        return *this;
    }

    DataMatrixHeap(T* mem, uint x_dim, uint y_dim)
        : DataMatrixView<T>()
    {
        this->x_size = x_dim;
        this->y_size = y_dim;

        this->data = new T[x_dim * y_dim];
        memcpy(this->data, mem, x_dim * y_dim * sizeof(T));
    }

    DataMatrixHeap(std::span<T> mem, uint x_dim, uint y_dim)
        : DataMatrixView<T>()
    {
        this->x_size = x_dim;
        this->y_size = y_dim;

        assert(mem.size() == x_dim * y_dim);

        this->data = new T[x_dim * y_dim];
        memcpy(this->data, mem.data(), x_dim * y_dim * sizeof(T));
    }

    ~DataMatrixHeap()
    {
        delete[] this->data;
    }

    operator DataMatrixView<T>() const
    {
        return DataMatrixView<T>{ this->data, this->x_size, this->y_size };
    }
};

struct Path
{
    std::vector<Vector2> points;
    float score = 0.0f;
    int score_rank = -1;

    DataMatrixHeap<float> dtw;
};

// Rescales path to fit in given rect
void AdjustPath(Rectangle rect, std::span<Vector2> path)
{
    Vector2 point_max = { FLT_MIN, FLT_MIN };
    Vector2 point_min = { FLT_MAX, FLT_MAX };

    for (uint i = 0; i < path.size(); i++)
    {
        point_max.x = std::max(path[i].x, point_max.x);
        point_max.y = std::max(path[i].y, point_max.y);

        point_min.x = std::min(path[i].x, point_min.x);
        point_min.y = std::min(path[i].y, point_min.y);
    }

    const int PADDING = 5;
    Vector2 dim{ point_max.x - point_min.x, point_max.y - point_min.y };

    for (Vector2& point : path)
    {
        point.x -= point_min.x;
        point.y -= point_min.y;

        if (fabsf(dim.x) > 0.01f)
        {
            point.x /= dim.x;
        }

        if (fabsf(dim.y) > 0.01f)
        {
            point.y /= dim.y;
        }

        point.x *= rect.width - (2 * PADDING);
        point.y *= rect.height - (2 * PADDING);

        point.x += rect.x + PADDING;
        point.y += rect.y + PADDING;
    }
}

int FindNearestPoint(std::span<Vector2> path, Vector2 pos, float* out_distance = nullptr)
{
    int min_idx = -1;
    float min_dst = FLT_MAX;

    for (uint i = 0; i < path.size(); i++)
    {
        float dist = Vector2Distance(path[i], pos);
        if (dist < min_dst)
        {
            min_dst = dist;
            min_idx = i;
        }
    }

    if (out_distance)
    {
        (*out_distance) = min_dst;
    }

    return min_idx;
}

bool RectangleContains(Rectangle rect, Vector2 point, float radius = 0.0f)
{
    if (rect.x - radius < point.x && rect.x + rect.width + radius > point.x)
    {
        if (rect.y - radius < point.y && rect.y + rect.height + radius > point.y)
        {
            return true;
        }
    }

    return false;
}

Rectangle RectFromPoints(Vector2 first, Vector2 second)
{
    Rectangle rect;
    rect.x = std::min(first.x, second.x);
    rect.y = std::min(first.y, second.y);
    rect.width = fabsf(first.x - second.x);
    rect.height = fabsf(first.y - second.y);

    return rect;
}

Rectangle RectangleInflate(const Rectangle& rect, float ratio)
{
    Rectangle result = rect;
    result.width *= ratio;
    result.height *= ratio;

    result.x -= 0.5f * (result.width - rect.width);
    result.y -= 0.5f * (result.height - rect.height);

    return result;
}

// --
// Dynamic Time Warping - ALGORITHM
int RankPathScore(std::vector<Path>& paths)
{
    int best_idx = -1;

    std::span<float> scores = STACK_ALLOC(float, paths.size());
    for (uint i = 0; i < paths.size(); i++)
    {
        scores[i] = paths[i].score;
    }
    std::sort(scores.begin(), scores.end());

    for (uint i = 0; i < paths.size(); i++)
    {
        for (uint j = 0; j < scores.size(); j++)
        {
            if (paths[i].score == scores[j])
            {
                paths[i].score_rank = j;

                // Save best path idx
                if (j == 0)
                {
                    best_idx = i;
                }
            }
        }
    }

    return best_idx;
}

// Window size [0.0f, 1.0f]
bool IsInWindow(int i, int j, int i_dim, int j_dim, float window)
{
    float i_span = i / float(i_dim);
    float j_span = j / float(j_dim);

    return fabsf(i_span - j_span) < window;
}

float CalcPathScoreDTW(std::span<Vector2> first, std::span<Vector2> second, float window, DataMatrixHeap<float>* out_dtw = nullptr)
{
    const int first_count = first.size();
    const int second_count = second.size();

    DataMatrixView<float> dtw = { STACK_ALLOC(float, uint(first_count * second_count)), uint(first_count), uint(second_count) };
    dtw.fill(FLT_MAX);
    dtw[0][0] = 0.0f;

    const auto distance = [&](uint i, uint j) {
        return Vector2Distance(first[i], second[j]);
    };

    // window = std::max(window, int(fabsf(first_count - second_count)));

    const auto is_in_window = [&](int i, int j) {
        return IsInWindow(i, j, first_count, second_count, window);
    };

    // Initialize window
    for (int i = 1; i < first_count; i++)
    {
        for (int j = 1; j < second_count; j++)
        {
            if (is_in_window(i, j))
            {
                dtw[j][i] = 0.0f;
            }
        }
    }

    // Compute
    for (int i = 1; i < first_count; i++)
    {
        for (int j = 1; j < second_count; j++)
        {
            if (is_in_window(i, j))
            {
                float cost = distance(i, j);

                float min_acc = std::min({
                    dtw[j][i - 1], // insertion
                    dtw[j - 1][i], // deletion
                    dtw[j - 1][i - 1]  // match
                });

                dtw[j][i] = cost + min_acc;
            }
        }
    }

    if (out_dtw)
    {
        (*out_dtw) = DataMatrixHeap<float>(dtw);
    }

    return dtw[second_count - 1][first_count - 1];
}


// -- 
// TOOLS

const float PATH_POINT_INTERACT_RADIUS = 10.0f;
const float PATH_POINT_DRAW_RADIUS = 7.5f;

const int BUTTON_PADDING = 50;
const int BUTTON_DIM = 40;

enum class EditPathState
{
    Idle,
    Move,
    RectSelect
};

struct EditPathCtx
{
    // Update each frame
    Vector2 mouse_pos;

    EditPathState state = EditPathState::Idle;

    float last_press_time = -1;
    int click_count = 0;

    Vector2 move_mouse_pos;
    Vector2 last_press_mouse_pos;
    Vector2 select_start_pos;
    std::vector<uint> selected_points;
};

enum EToolbarSide
{
    Left,
    Right
};

struct ToolbarDesc
{
    EToolbarSide side = EToolbarSide::Left;
    Color color = RED;
    std::span<Vector2> default_path = {};
};

void HandleEditPathViewport(EditPathCtx& ctx, Path& path)
{
    const auto find_hover_point = [&]() -> int {
        for (uint i = 0; i < path.points.size(); i++)
        {
            if (Vector2Distance(path.points[i], ctx.mouse_pos) < PATH_POINT_INTERACT_RADIUS)
            {
                return i;
            }
        }

        return -1;
    };

    switch (ctx.state)
    {
        case EditPathState::RectSelect:
        {
            if (IsMouseButtonReleased(0))
            {
                ctx.state = EditPathState::Idle;
            }
            else
            {
                const Rectangle select_rect = RectFromPoints(ctx.select_start_pos, ctx.mouse_pos);

                ctx.selected_points.clear();

                for (uint i = 0; i < path.points.size(); i++)
                {
                    if (RectangleContains(select_rect, path.points[i], PATH_POINT_INTERACT_RADIUS))
                    {
                        ctx.selected_points.push_back(i);
                    }
                }
            }

            break;
        }
        case EditPathState::Idle:
        {
            const float DOUBLE_CLICK_TIME = 0.2f;
            const float HOLD_TIME = 0.05f;

            if (GetTime() - ctx.last_press_time > DOUBLE_CLICK_TIME)
            {
                ctx.click_count = 0;
            }

            if (IsMouseButtonPressed(0))
            {
                ctx.click_count++;
                ctx.last_press_time = GetTime();
                ctx.last_press_mouse_pos = ctx.mouse_pos;
            }

            if (!ctx.selected_points.empty())
            {
                if (IsKeyPressed(KEY_DELETE) || IsKeyPressed(KEY_BACKSPACE))
                {
                    RemoveIndicesFromVector(path.points, ctx.selected_points);
                    ctx.selected_points.clear();
                }
                else if (IsMouseButtonDown(0))
                {
                    for (uint i : ctx.selected_points)
                    {
                        if (Vector2Distance(path.points[i], ctx.mouse_pos) < PATH_POINT_INTERACT_RADIUS)
                        {
                            ctx.state = EditPathState::Move;
                            ctx.move_mouse_pos = ctx.mouse_pos;
                            return;
                        }
                    }

                    ctx.selected_points.clear();
                }

                break;
            }

            if (IsMouseButtonDown(0) && GetTime() - ctx.last_press_time > HOLD_TIME)
            {
                // Handle select and move single point
                if (int hover_point_idx = find_hover_point(); hover_point_idx != -1)
                {
                    if (IndexOf<uint>(ctx.selected_points, hover_point_idx) == -1)
                    {
                        ctx.selected_points.push_back(hover_point_idx);
                    }

                    ctx.state = EditPathState::Move;
                    ctx.move_mouse_pos = ctx.last_press_mouse_pos;
                }
                else
                {
                    ctx.state = EditPathState::RectSelect;
                    ctx.select_start_pos = ctx.last_press_mouse_pos;
                }
            }
            else if (IsMouseButtonPressed(0))
            {
                // Insert new point on double click
                if (ctx.click_count > 1)
                {
                    int idx = FindNearestPoint(path.points, ctx.mouse_pos);

                    // last point
                    if (idx == path.points.size() - 1)
                    {
                        Vector2 dir_to_next = Vector2Subtract(path.points[idx - 1], path.points[idx]);
                        Vector2 dir_to_new = Vector2Subtract(ctx.mouse_pos, path.points[idx]);

                        if (Vector2DotProduct(dir_to_next, dir_to_new) < 0)
                        {
                            // insert after
                            path.points.insert(path.points.begin() + idx + 1, ctx.mouse_pos);
                        }
                        else
                        {
                            // insert before
                            path.points.insert(path.points.begin() + idx, ctx.mouse_pos);
                        }
                    }
                    else
                    {
                        Vector2 dir_to_next = Vector2Subtract(path.points[idx + 1], path.points[idx]);
                        Vector2 dir_to_new = Vector2Subtract(ctx.mouse_pos, path.points[idx]);

                        if (Vector2DotProduct(dir_to_next, dir_to_new) > 0)
                        {
                            // insert after
                            path.points.insert(path.points.begin() + idx + 1, ctx.mouse_pos);
                        }
                        else
                        {
                            // insert before
                            path.points.insert(path.points.begin() + idx, ctx.mouse_pos);
                        }
                    }
                }
                else
                {
                    if (int hover_point_idx = find_hover_point(); hover_point_idx != -1)
                    {
                        if (IndexOf<uint>(ctx.selected_points, hover_point_idx) == -1)
                        {
                            ctx.selected_points.push_back(hover_point_idx);
                        }

                        ctx.state = EditPathState::Move;
                        ctx.move_mouse_pos = ctx.mouse_pos;
                    }
                }
            }

            break;
        }
        case EditPathState::Move:
        {
            Vector2 mouse_delta = Vector2Subtract(ctx.mouse_pos, ctx.move_mouse_pos);

            for (uint i : ctx.selected_points)
            {
                path.points[i] = Vector2Add(path.points[i], mouse_delta);
            }

            ctx.move_mouse_pos = ctx.mouse_pos;

            if (IsMouseButtonReleased(0))
            {
                ctx.state = EditPathState::Idle;
            }

            break;
        }
    }
}

void DrawPath(std::span<Vector2> path, Color color = RED, float thickness = 1.5f)
{
    // Lines
    for (int i = 1; i < path.size(); i++)
    {
        DrawLineEx(path[i - 1], path[i], thickness, color);
    }

    // Points
    for (int i = 0; i < path.size(); i++)
    {
        DrawCircleV(path[i], PATH_POINT_DRAW_RADIUS, color);
    }
}

void DrawEditPathViewport(EditPathCtx& ctx, std::span<Vector2> path, Color color = RED, float thickness = 1.5f)
{
    if (ctx.state == EditPathState::RectSelect)
    {
        Rectangle select_rect = RectFromPoints(ctx.select_start_pos, ctx.mouse_pos);

        DrawRectangleRec(select_rect, ColorAlpha(ColorBrightness(SKYBLUE, 0.7f), 0.5f));
        DrawRectangleLinesEx(select_rect, 1.0f, ColorBrightness(SKYBLUE, 0.5f));
    }

    // Lines
    for (int i = 1; i < path.size(); i++)
    {
        DrawLineEx(path[i - 1], path[i], thickness, color);
    }

    auto is_selected = [&](uint i)
    {
        return IndexOf<uint>(ctx.selected_points, i) != -1;
    };

    // Points
    for (int i = 0; i < path.size(); i++)
    {
        if (is_selected(i))
        {
            switch (ctx.state)
            {
                case EditPathState::RectSelect:
                case EditPathState::Idle:
                {
                    DrawCircleLinesV(path[i], PATH_POINT_DRAW_RADIUS * 1.2f, BLACK);
                    DrawCircleV(path[i], PATH_POINT_DRAW_RADIUS, ColorBrightness(color, 0.5f));
                    break;
                }
                case EditPathState::Move:
                {
                    DrawCircleLinesV(path[i], PATH_POINT_DRAW_RADIUS * 1.2f, BLACK);
                    DrawCircleV(path[i], PATH_POINT_DRAW_RADIUS, ColorBrightness(color, -0.25f));
                    break;
                }
            }
        }
        else
        {
            DrawCircleV(path[i], PATH_POINT_DRAW_RADIUS, color);
        }
    }

    const int MOUSE_LABEL_PADDING = 10;

    // Mouse label
    switch (ctx.state)
    {
    //case EditPathState::Pick:
    //{
    //    DrawText("Pick", ctx.mouse_pos.x, ctx.mouse_pos.y, 5, DARKGRAY);
    //    break;
    //}
    case EditPathState::Move:
    {
        DrawText("Move", ctx.mouse_pos.x + MOUSE_LABEL_PADDING, ctx.mouse_pos.y + MOUSE_LABEL_PADDING, 5, DARKBLUE);
        break;
    }
    default:
        break;
    }
}

void DrawPathThumbnail(Rectangle rect, std::span<Vector2> path, float thickness = 1.0f, Color color = RED)
{
    std::span<Vector2> thumb_path = STACK_ALLOC(Vector2, path.size());
    std::copy(path.begin(), path.end(), thumb_path.begin());

    AdjustPath(rect, thumb_path);

    // Draw
    for (int i = 1; i < thumb_path.size(); i++)
    {
        DrawLineEx(thumb_path[i - 1], thumb_path[i], thickness, color);
    }
}

bool GuiEditPathToolbar(Vector2 start_pos, const char* label, std::vector<Path>& paths, int& current_path_idx, ToolbarDesc desc = {})
{
    bool act = false;

    const int text_width = GetTextWidth(label);
    const int text_pos_x = start_pos.x + (BUTTON_DIM / 2) - (text_width / 2);

    const int side_mul = desc.side == EToolbarSide::Left ? 1 : -1;

    DrawText(label, text_pos_x, start_pos.y, 5, GRAY);

    Rectangle button_rect = { start_pos.x, start_pos.y + 20, BUTTON_DIM, BUTTON_DIM };
    const Rectangle button_origin = button_rect;

    for (uint i = 0; i < paths.size(); i++)
    {
        if (i == current_path_idx)
        {
            DrawRectangleLinesEx(RectangleInflate(button_rect, 1.25f), 2.0f, BLACK);
        }

        if (GuiButton(button_rect, ""))
        {
            current_path_idx = i;
            act = true;
        }

        button_rect.y += BUTTON_PADDING;
    }

    if (current_path_idx != -1)
    {
        Rectangle cmds_rect = button_origin;
        cmds_rect.x += BUTTON_PADDING * side_mul;
        cmds_rect.y += BUTTON_PADDING * current_path_idx;

        if (GuiButton(cmds_rect, "DEL"))
        {
            paths.erase(paths.begin() + current_path_idx);
            
            if (current_path_idx >= paths.size())
            {
                current_path_idx--;
            }

            act = true;
        }

        cmds_rect.x += BUTTON_PADDING * side_mul;
    }

    if (GuiButton(button_rect, "NEW"))
    {
        current_path_idx = paths.size();
        paths.emplace_back(Path{ .points = CopySpan(desc.default_path) });

        act = true;
    }

    // thumbnails
    Rectangle thumbnail_rect = button_origin;
    for (uint i = 0; i < paths.size(); i++)
    {
        DrawPathThumbnail(thumbnail_rect, paths[i].points, 2.0f, desc.color);

        if (paths[i].score == FLT_MAX)
        {
            int txt_width = GetTextWidth("INF");

            DrawText
            (
                "INF", 
                thumbnail_rect.x + (thumbnail_rect.width / 2) - (txt_width / 2),
                thumbnail_rect.y + (thumbnail_rect.height / 3),
                5, BLACK
            );
        }
        else if (paths[i].score != 0.0f)
        {
            std::string label = std::format("{}", paths[i].score_rank);
            const int txt_width = GetTextWidth(label.c_str());

            DrawText
            (
                label.c_str(), 
                thumbnail_rect.x + (thumbnail_rect.width / 2) - (txt_width / 2),
                thumbnail_rect.y + (thumbnail_rect.height / 3),
                20, BLACK
            );
        }

        thumbnail_rect.y += BUTTON_PADDING;
    }

    return act;
}

void DrawMatrix(Rectangle rect, DataMatrixView<float> matrix, float min_val, float max_val, Color val_color, float window)
{
    Vector2 begin_pos = { rect.x, rect.y };

    int x_cell_size = rect.width / matrix.x_size;
    int y_cell_size = rect.height / matrix.y_size;

    float val_range = max_val - min_val;

    int y_text_pad = y_cell_size / 3;
    int x_text_pad = 5;

    for (uint i = 0; i < matrix.x_size; i++)
    {
        for (uint j = 0; j < matrix.y_size; j++)
        {
            float val = (matrix[j][i] - min_val) / val_range;

            Vector2 pos = begin_pos;
            pos.x += i * x_cell_size;
            pos.y += j * y_cell_size;

            bool in_window = IsInWindow(i, j, matrix.x_size, matrix.y_size, window);

            Rectangle rec = { pos.x, pos.y, x_cell_size, y_cell_size };

            if (matrix[j][i] == FLT_MAX)
            {
                DrawRectangleRec(rec, ColorBrightness(val_color, -0.5f));
                DrawText("INF",
                    x_text_pad + pos.x, y_text_pad + pos.y, 4, LIGHTGRAY);
            }
            else
            {
                DrawRectangleRec(rec, ColorAlpha(val_color, val));
                DrawText(std::format("{:.2f}", matrix[j][i]).c_str(), 
                    x_text_pad + pos.x, y_text_pad + pos.y, 4, BLACK);
            }

            if (in_window)
            {
                DrawRectangleLinesEx(rec, 2.0f, SKYBLUE);
            }
        }
    }
}

// --

int main()
{
    // Create Window
    const int screenWidth = 800;
    const int screenHeight = 450;

    SetWindowState(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);

    InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

    SetTargetFPS(60);

    // Defaults
    const Color ANIM_PATH_COLOR = RED;
    const Color TRACK_PATH_COLOR = SKYBLUE;

    std::vector<Vector2> default_anim_path = {
        { -50, -50 }, // act start
        { -50, -25 }, // phase start
        { -50,   0 }, // corr start
        { -38,  27 }, // corr mid
        {   0,  50 }, // corr end
        {  25,  50 }, // phase end
        {  50,  50 }, // final
    };

    std::vector<Vector2> default_track_path = {
        { -50, -50 },
        {   0,   0,},
        {  50,  50 }
    };

    // Move default paths to center of the screen
    for (Vector2& vec : default_anim_path)
    {
        vec.x += screenWidth / 2;
        vec.y += screenHeight / 2;
    }

    for (Vector2& vec : default_track_path)
    {
        vec.x += screenWidth / 2;
        vec.y += screenHeight / 2;
    }

    // Main data
    std::vector<Path> anim_paths;
    std::vector<Path> track_paths;

    EditPathCtx edit_tool;

    int current_edit_anim_path = -1;
    int current_edit_track_path = -1;
    int last_edit_track_path = -1;
    bool show_matrix = false;

    float WINDOW_COEFF = 0.3f;

    // Main game loop
    while (!WindowShouldClose())
    {
        // --
        // Update path edit tool

        Vector2 mouse_pos = GetMousePosition();
        edit_tool.mouse_pos = mouse_pos;

        if (current_edit_anim_path != -1)
        {
            HandleEditPathViewport(edit_tool, anim_paths[current_edit_anim_path]);
        }

        if (current_edit_track_path != -1)
        {
            HandleEditPathViewport(edit_tool, track_paths[current_edit_track_path]);
            last_edit_track_path = current_edit_track_path;
        }

        // --
        // Draw

        BeginDrawing();
        {
            ClearBackground(RAYWHITE);

            // Draw order
            // - anim paths not currently in edit
            // - either currently edited track path or last selected track path 
            // - currently edited anim path

            // Draw anim paths not currently in edit in gray
            for (uint i = 0; i < anim_paths.size(); i++)
            {
                if (current_edit_anim_path != i)
                {
                    DrawPath(anim_paths[i].points, LIGHTGRAY);
                }
            }

            // Draw edited track path tool
            if (current_edit_track_path != -1)
            {
                DrawEditPathViewport(edit_tool, track_paths[current_edit_track_path].points, TRACK_PATH_COLOR, 2.0f);
            }
            // Or draw last selected track path when not in edit in desaturated blue
            else if (last_edit_track_path != -1 && last_edit_track_path < track_paths.size())
            {
                DrawPath(track_paths[last_edit_track_path].points, ColorBrightness(ColorTint(TRACK_PATH_COLOR, LIGHTGRAY), 0.5f), 2.0f);
            }

            // Draw edited anim path tool
            if (current_edit_anim_path != -1)
            {
                DrawEditPathViewport(edit_tool, anim_paths[current_edit_anim_path].points, ANIM_PATH_COLOR);
            }

            // --
            // GUI - semi update / draw
            GuiEditPathToolbar
            (
                { 15, 10 }, "ANIM", anim_paths, current_edit_anim_path,
                { .color = ANIM_PATH_COLOR, .default_path = default_anim_path }
            );

            // Only one track editable at once
            // When anim path was edited, reset track path
            if (current_edit_anim_path != -1)
            {
                current_edit_track_path = -1;
            }

            // --

            GuiEditPathToolbar
            (
                { screenWidth - BUTTON_DIM - 15, 10 }, "TRACK", track_paths, current_edit_track_path,
                { .side = EToolbarSide::Right, .color = TRACK_PATH_COLOR, .default_path = default_track_path }
            );

            // Only one track editable at once
            // When track path was edited, reset anim path
            if (current_edit_track_path != -1)
            {
                current_edit_anim_path = -1;
            }

            // -- 
            // Gui to set DTW window size and enable debug matrix rendering

            const Vector2 WINDOW_SLIDER_SIZE = { 100, 20 };
            Rectangle dtw_window_slider_rect =
            {
                (screenWidth / 2) - (WINDOW_SLIDER_SIZE.x / 2), screenHeight * 0.85f,
                WINDOW_SLIDER_SIZE.x, WINDOW_SLIDER_SIZE.y
            };

            const Vector2 DTW_CHECK_SIZE = { 20, 20 };
            Rectangle dtw_debug_checkbox_rect =
            { 
                dtw_window_slider_rect.x, dtw_window_slider_rect.y + 10 + WINDOW_SLIDER_SIZE.y,
                DTW_CHECK_SIZE.x, DTW_CHECK_SIZE.y
            };

            Rectangle dtw_window_label_rect = dtw_window_slider_rect;
            dtw_window_label_rect.y -= 2 * WINDOW_SLIDER_SIZE.y - 10;
            dtw_window_label_rect.width *= 1.5f;

            GuiSliderBar(dtw_window_slider_rect, "0.0f", "1.0f", &WINDOW_COEFF, 0.0f, 1.0f);
            GuiLabel(dtw_window_label_rect, std::format("Window coeff: {:.4f}", WINDOW_COEFF).c_str());
            GuiCheckBox(dtw_debug_checkbox_rect, "Show DTW matrix", &show_matrix);

            // --
            // MAIN ALGORITHM

            // Solve DTW
            if (!anim_paths.empty() && last_edit_track_path != -1)
            {
                Path& track_path = track_paths[last_edit_track_path];
                for (uint i = 0; i < anim_paths.size(); i++)
                {

                    anim_paths[i].score = CalcPathScoreDTW
                    (
                        track_path.points,
                        anim_paths[i].points,
                        WINDOW_COEFF,
                        &anim_paths[i].dtw
                    );
                }
            }

            // Rank paths
            int best_path_idx = RankPathScore(anim_paths);

            // Draw debug matrix if applicable
            const int MATRIX_VIEW_SIZE_X = 300;
            const int MATRIX_VIEW_SIZE_Y = 150;

            if (show_matrix && (current_edit_anim_path != -1 || best_path_idx != -1))
            {
                // Pick either currently edited anim path, or the best one when authoring track path
                int path_idx = current_edit_anim_path != -1 ? current_edit_anim_path : best_path_idx;

                DrawMatrix
                (
                    Rectangle{ 10, screenHeight - 10 - MATRIX_VIEW_SIZE_Y, MATRIX_VIEW_SIZE_X, MATRIX_VIEW_SIZE_Y },
                    anim_paths[path_idx].dtw,
                    0.0f,
                    anim_paths[path_idx].score,
                    RED,
                    WINDOW_COEFF
                );
            }
        }
        EndDrawing();
    }

    // Clean-up
    CloseWindow();

	return 0;
}