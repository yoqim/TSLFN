


def cam(feats, bboxes_list):
    """
    feats: list, [Tensor(bs x c_l x h_l x w_j), Tensor, Tensor...], Feature maps produced by FPN
    bboxes_list: list, [Tensor(n x 5), Tensor(n x 5) ...], len(bboxes_list) = batchsize
    """
    # 1. Get init classification result
    boxes = [torch.narrow(bboxes, dim=1, start=0, length=4) for bboxes in bboxes_list]
    pooled_feats = roi_heads.pooler(feats, boxes)
    
    cls_logits, _ = roi_heads.box_predictor(roi_heads.box_head(pooled_feats))
    probs = torch.softmax(cls_logits, dim=-1)
    # Remove last column (background)
    probs = torch.narrow(probs, dim=1, start=0, length=roi_heads.num_classes)
    _, cls = torch.max(probs, dim=1)

    # 2. Channel-wise self-attention
    bs, c = pooled_feats.shape[:2]
    feats = pooled_feats
    relu_feats = F.relu(feats)

    feats = feats.unsqueeze(1)
    relu_feats = relu_feats.unsqueeze(2)

    # bs x c x c x h x w
    att_feats = feats * relu_feats
    # (bs x c x c) x h x w
    att_feats = att_feats.view(-1, *att_feats.shape[2:])

    # 3. Get score of all the feature groups
    cls_logits, _ = roi_heads.box_predictor(roi_heads.box_head(att_feats))
    probs = torch.softmax(cls_logits, dim=-1)

    probs = probs.view(bs, c, -1).permute(0, 2, 1)  # bs, num_cls, c
    weight = probs[torch.arange(0, cls.shape[0]), cls].unsqueeze(2).unsqueeze(2)

    # 4. Weighted sum
    cam = (pooled_feats * weight).sum(dim=1)
    cam = F.relu(cam)

    # 5. Norm
    cam_ = cam.view(cam.shape[0], -1)
    min_v, _ = cam_.min(dim=1)
    cam_ -= min_v.view(-1, 1)
    max_v, _ = cam_.max(dim=1)
    cam_ /= max_v.view(-1, 1) + 1e-15
    cam = cam_.view(-1, *feats.shape[2:])

    num = [box.shape[0] for box in boxes]
    cams = cam.split(num)
    return cams



