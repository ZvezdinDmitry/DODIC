import time

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.concurrency import run_in_threadpool

from api.images_processing import fig_to_base64
from api.models_inference import get_model

APP_STATS = {"total_requests": 0, "total_time": 0.0}
MAX_FILE_SIZE = 1 * 1024 * 1024

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"))

templates = Jinja2Templates(directory="templates")

instrumentator = Instrumentator(
    should_group_status_codes=False,
    # should_ignore_untargeted=True,
)

# instrumentator.add(metrics.request_latency_histograms())
# instrumentator.add(metrics.request_size_bytes_histograms())
# instrumentator.add(metrics.response_size_bytes_histograms())
# instrumentator.add(
#     metrics.requests_in_progress(
#         metric_name="http_requests_in_progress", labels={"handler": "handler"}
#     )
# )

instrumentator.instrument(app).expose(app, endpoint="/metrics")


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request) -> Response:
    return templates.TemplateResponse(request, "index.html")


# test of model: damages in damaged and vice verse
# mb add pure api section (think about routing) + pydantic
@app.post("/", response_class=HTMLResponse)
async def infer_model(
    request: Request,
    file: UploadFile = File(...),
    model=Depends(get_model, use_cache=True),
    damage_conf: float = Form(0.30),
    part_conf: float = Form(0.20),
    part_overlap: float = Form(0.2),
) -> Response:
    ctx: dict = {}
    start_time = time.time()
    try:
        damages = []
        file_bytes = await file.read()
        file_size = len(file_bytes)

        if file_size == 0:
            ctx.update(
                error="Файл не был отправлен или сессия истекла. Пожалуйста, выберите изображение заново."
            )
            return templates.TemplateResponse(
                request,
                "index.html",
                ctx,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if file_size > MAX_FILE_SIZE:
            ctx.update(
                error=f"Файл слишком большой ({round(file_size / (1024*1024), 2)} МБ). Максимальный размер — {MAX_FILE_SIZE // (1024*1024)} МБ."
            )
            return templates.TemplateResponse(
                request,
                "index.html",
                ctx,
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
            # raise HTTPException(
            #     status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            #     detail=,
            # )

        image = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            ctx.update(
                error="Не удалось распознать формат изображения. Пожалуйста, загрузите валидный JPG/PNG."
            )
            return templates.TemplateResponse(
                request,
                "index.html",
                ctx,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        damages_img, mapped_damages = await run_in_threadpool(
            model.annotate, image, part_overlap, part_conf, damage_conf
        )

        for damage_type, part, confidence in mapped_damages:
            damages.append(
                {
                    "damage_type": damage_type,
                    "part": part,
                    "confidence": confidence,
                }
            )

        current_request_time = time.time() - start_time
        APP_STATS["total_requests"] += 1
        APP_STATS["total_time"] += current_request_time
        average_request_time = (
            APP_STATS["total_time"] / APP_STATS["total_requests"]
        )
        base64_image = await run_in_threadpool(fig_to_base64, damages_img)
        ctx.update(
            image=base64_image,
            damages=damages,
            current_time=round(current_request_time, 3),
            average_time=round(average_request_time, 3),
            total_requests=APP_STATS["total_requests"],
        )

    except Exception as err:
        ctx.update(
            error=f"Внутренняя ошибка сервера при обработке модели: {type(err).__name__}"
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            ctx,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return templates.TemplateResponse(request, "index.html", ctx)
