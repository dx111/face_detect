from fastapi import FastAPI, Depends, status
from pydantic import BaseModel, Field
import numpy
import cv2
import numpy as np
from typing import List, Any, Dict
import base64
from centerface import CenterFace
from starlette.exceptions import HTTPException
from starlette.requests import Request
from fastapi.responses import JSONResponse
import exceptions
import datetime

description = """
人脸检测的接口
"""

app = FastAPI(
    title="人脸检测",
    description=description,
    version="v0.0.1",)

class request(BaseModel):
    # 图片base64编码
    imageBase64: str = Field(..., example="")

class face_detect_struct(BaseModel):
    # 置信度
    score: float = Field(..., example=0.88)
    # 人脸框
    box: List[int] = Field(..., example=[0, 0, 100, 100])
    # 五个人脸关键点
    landmark: List[List[int]] = Field(..., example=[[0, 0], [100, 0], [150, 150], [300, 0], [300, 300]])


# 返回值
class face_detect_response(BaseModel):
    face_count:float=Field(...,example=1)
    face_info: List[face_detect_struct]

# def get_detect_inference():
#     landmarks = True
#     centerface = CenterFace(landmarks=landmarks)
#     def inference(frame):
#         h, w = frame.shape[:2]
#         dets, lms = centerface(frame, h, w, threshold=0.35)
#         return dets,lms
    
#     return inference

# inference=get_detect_inference()




@app.post("/face_detect",
          summary="人脸检测",
          response_model=face_detect_response)
def face_detect(req: request):
    try:
        imgData = base64.b64decode(req.imageBase64)
    except:
        raise exceptions.DecodeError("decode imageBase64 error")
    nparr = np.fromstring(imgData, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        pass
    h,w=frame.shape[:2]

    centerface = CenterFace(landmarks=True)
    dets, lms = centerface(frame, h, w, threshold=0.35)
    # dets, lms=inference(frame)
    face_detect_list = []
    for det, lm in zip(dets, lms):
        boxes, score = det[:4], det[4]
        landmark = []
        for i in range(0, 5):
            x = int(lm[i * 2])
            y = int(lm[i * 2 + 1])
            landmark.append([x, y])
        box_info = {
            "score": float(score),
            "box": [int(boxes[0]), int(boxes[1]), int(boxes[2]-boxes[0]), int(boxes[3]-boxes[1])],
            "landmark": landmark
        }
        face_detect_list.append(box_info)
    result={"face_count":len(face_detect_list),"face_info":face_detect_list}
    return face_detect_response(**result)


@app.exception_handler(HTTPException)
async def exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"isSuc": False, "code": -1, "msg": "Not Found", "res": ""}
    )


@app.exception_handler(exceptions.DecodeError)
async def decode_exception_handler(request: Request, exc: exceptions.DecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"isSuc": False, "code": -1, "msg": exc.message, "res": ""}
    )
