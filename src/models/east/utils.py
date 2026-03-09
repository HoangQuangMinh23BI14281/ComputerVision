import os
import json
import math
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_annotation(annotation_path):
    boxes = list()

    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            line = line.split(',')

            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            x3 = line[4]
            y3 = line[5]
            x4 = line[6]
            y4 = line[7]

            cor = [x1, y1, x2, y2, x3, y3, x4, y4]
            boxes.append(cor)

    return boxes


def create_json_data(directory='data/raw_data/train', images_json='EAST/data/images.json',
                     boxes_json='EAST/data/boxes.json'):
    image_paths = list()
    boxes = list()

    for filename in os.listdir(directory):
        filename = filename[: -4]

        if filename + '.txt' not in os.listdir(directory):
            continue

        if filename + '.jpg' not in os.listdir(directory):
            continue

        image_paths.append(os.path.join(directory, filename + '.jpg'))
        boxes.append(parse_annotation(directory + '/' + filename + '.txt'))

    with open(images_json, 'w') as f:
        json.dump(image_paths, f)

    with open(boxes_json, 'w') as f:
        json.dump(boxes, f)


def cal_distance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.2):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):

    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_map = get_rotate_mat(theta)
    res = np.dot(rotate_map, v - anchor)

    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices

    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)

    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
            cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)

    return err


def find_min_rect_angle(vertices):
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                    np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y



def rotate_img(img, vertices, angle_range=10):
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices


def resize(img, vertices, length):
    shape = img.size
    new_image = img.resize((length, length))

    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
      new_vertices[i, [0, 2, 4, 6]] = vertices[i, [0, 2, 4, 6]] * (length / shape[0])
      new_vertices[i, [1, 3, 5, 7]] = vertices[i, [1, 3, 5, 7]] * (length / shape[1])
    
    return new_image, new_vertices


def get_score_geo(img, vertices, scale, length):
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)

    # Vectorized coordinate computation
    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    
    # Precompute coordinate matrix for rotation once
    x_lin = index_x.reshape((1, index_x.size))
    y_lin = index_y.reshape((1, index_x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)

    polys = []
    for i, vertice in enumerate(vertices):
        # Shrunk polygon for score map
        # Reduced coef to 0.2 to avoid disappearing small text
        shrunk_v = shrink_poly(vertice, coef=0.2)
        poly = np.around(scale * shrunk_v.reshape((4, 2))).astype(np.int32)
        
        # Ensure the shrunk polygon has at least some area
        if cv2.contourArea(poly) < 1:
            # Fallback: use slightly less shrunk version if it disappeared
            shrunk_v = shrink_poly(vertice, coef=0.1)
            poly = np.around(scale * shrunk_v.reshape((4, 2))).astype(np.int32)
            
        polys.append(poly)
        
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)
        
        # Binary mask for active indices to speed up geo_map updates
        mask_binary = (temp_mask > 0)
        if not np.any(mask_binary):
            # Last resort: if still no pixels, the text is too small for this resolution
            continue

        # Calculate actual angle for the box
        theta = find_min_rect_angle(vertice)
        
        # Optimized geometry calculation
        # Rotate back to axis-aligned for d1-d4 calculation
        # Note: we use -theta because we want to align the rotated box to axes
        rotated_vertices = rotate_vertices(vertice, -theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        
        # Only compute for masked pixels
        active_coords = coord_mat[:, mask_binary.flatten()]
        
        # For geometry calculation, we also need to rotate the PIXEL coordinates 
        # to the same coordinate system as the axis-aligned box
        # But for d1-d4 relative to the rotated box, we can just rotate the pixels
        # around the same anchor used in rotate_vertices
        anchor = vertice.reshape((4, 2)).T[:, :1]
        rotate_map = get_rotate_mat(-theta)
        active_coords_rotated = np.dot(rotate_map, active_coords - anchor) + anchor

        # d1: top, d2: bottom, d3: left, d4: right
        geo_map[mask_binary, 0] = active_coords_rotated[1] - y_min
        geo_map[mask_binary, 1] = y_max - active_coords_rotated[1]
        geo_map[mask_binary, 2] = active_coords_rotated[0] - x_min
        geo_map[mask_binary, 3] = x_max - active_coords_rotated[0]
        geo_map[mask_binary, 4] = theta

    cv2.fillPoly(score_map, polys, 1)
    score_map = torch.Tensor(score_map).permute(2, 0, 1)
    geo_map = torch.Tensor(geo_map).permute(2, 0, 1)

    return score_map, geo_map

def is_valid_poly(res, score_shape, scale):
    """Checks if a polygon is within image boundaries."""
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
           res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False

def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :] # d1, d2, d3, d4
    theta = valid_geo[4, :] # angle
    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        t = theta[i]
        
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        
        # Axis-aligned relative to the rotation anchor
        # In get_score_geo, we used active_coords_rotated = map(active) + anchor
        # So here we do the reverse
        coord = np.array([[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]])
        
        # Rotate back by theta
        rotate_mat = get_rotate_mat(t)
        # Using (x,y) of the pixel as the anchor for rotation restoration
        # (This matches how we calculated active_coords_rotated)
        anchor = np.array([[x], [y]])
        coord_rotated = np.dot(rotate_mat, coord - anchor) + anchor
        
        if is_valid_poly(coord_rotated, score_shape, scale):
            # Geometry Sanity Check: Filter out "wild" boxes
            # Box shouldn't be larger than the score map effectively
            h_map, w_map = score_shape
            box_w = np.linalg.norm(coord_rotated[:, 0] - coord_rotated[:, 1])
            box_h = np.linalg.norm(coord_rotated[:, 0] - coord_rotated[:, 3])
            
            # If box is extremely large (e.g. > 80% image size) in one dimension while tiny in other
            # it's likely noise.
            if box_w > w_map * scale * 0.9 or box_h > h_map * scale * 0.9:
                if box_w / (box_h + 1e-5) > 50 or box_h / (box_w + 1e-5) > 50:
                    continue

            index.append(i)
            # Order: x1, y1, x2, y2, x3, y3, x4, y4
            polys.append([coord_rotated[0, 0], coord_rotated[1, 0], coord_rotated[0, 1], coord_rotated[1, 1],
                         coord_rotated[0, 2], coord_rotated[1, 2], coord_rotated[0, 3], coord_rotated[1, 3]])
    return np.array(polys), index

def py_nms(dets, thresh):
    """
    Improved Python NMS fallback.
    Calculates the true axis-aligned bounding box of the rotated polygon 
    for more accurate overlap calculation.
    """
    if dets.shape[0] == 0: return []
    
    # dets format: x1, y1, x2, y2, x3, y3, x4, y4, score
    polys = dets[:, :8].reshape(-1, 4, 2)
    scores = dets[:, 8]
    
    # Calculate true axis-aligned bounding boxes (min/max of all 4 points)
    x1 = np.min(polys[:, :, 0], axis=1)
    y1 = np.min(polys[:, :, 1], axis=1)
    x2 = np.max(polys[:, :, 0], axis=1)
    y2 = np.max(polys[:, :, 1], axis=1)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def get_east_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """Converts model score/geo maps to final bounding boxes."""
    if len(score.shape) == 4: # Handle [N, C, H, W]
        score = score[0]
        geo = geo[0]
        
    score = score[0, :, :].cpu().numpy() if isinstance(score, torch.Tensor) else score[0, :, :]
    geo = geo.cpu().numpy() if isinstance(geo, torch.Tensor) else geo
    
    xy_text = np.argwhere(score > score_thresh)
    if xy_text.size == 0: return np.array([])
    
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    
    if polys_restored.size == 0: return np.array([])
    
    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    
    try:
        try:
            import lanms
            boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
        except (ImportError, AttributeError):
            # Try lanms-nova which is often used as a drop-in or specialized fork
            import lanms_nova
            boxes = lanms_nova.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    except ImportError:
        # Use proper Python NMS fallback (which uses torchvision if available)
        keep = py_nms(boxes, nms_thresh)
        boxes = boxes[keep]
        
    return boxes

if __name__ == '__main__':
    pass
