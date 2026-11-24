import sys
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Arcface.ranking import rank_faces
from RetinaFace.detect_crop import detect_and_crop
from SHQS.shqs_opencv import shqs_score, level_of
from SHQS.stage2_clahe import clahe_enhance

OUTPUT = ROOT / "Pipeline_Output"


def run_pipeline(image_path):

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not load input image")

    output = OUTPUT / image_path.stem

    for folder in ["raw", "no", "soft", "strong", "low_conf"]:
        (output / folder).mkdir(parents=True, exist_ok=True)

    # STAGE 1: RETINAFACE
    print("\n===== RETINAFACE =====")

    faces = detect_and_crop(image)

    print("Total faces detected:", len(faces))

    report = []

    for face in faces:

        face_id = face["face_id"]
        crop = face["crop"]
        name = f"face_{face_id}.png"

        print(f"\n===== FACE {face_id} =====")

        entry = {
            "face_id": face_id,
            "detector_score": face["detector_score"],
            "box": face["box"],
            "landmarks": face["landmarks"],
        }

        if not face["accepted"]:

            cv2.imwrite(str(output / "low_conf" / name), crop)

            entry["status"] = "LOW_CONFIDENCE"
            report.append(entry)

            print("Low confidence detection")
            continue

        cv2.imwrite(str(output / "raw" / name), crop)

        # STAGE 2: SHQS
        score, factors, found, measurements = shqs_score(crop)

        decision = level_of(score)

        print(f"SHQS Score: {score:.2f}")
        print("Decision:", decision)

        # STAGE 3: SELECTIVE ENHANCEMENT

        if decision == "NO":

            processed = crop
            method = "Original"
            status = "READY_FOR_ARCFACE"

            destination = output / "no" / name

        elif decision == "SOFT":

            processed = clahe_enhance(crop)
            method = "CLAHE"
            status = "READY_FOR_ARCFACE"

            destination = output / "soft" / name

        else:

            processed = crop
            method = "CodeFormer"
            status = "CODEFORMER_PENDING"

            destination = output / "strong" / name

        cv2.imwrite(str(destination), processed)

        # STAGE 4: ARCFACE RECOGNITION

        recognition_result = None

        if decision in ["NO", "SOFT"]:

            print("\n===== ARCFACE =====")

            recognition_result = rank_faces(processed, top_k=5, threshold=0.5)

            status = "RECOGNITION_COMPLETED"

            print("Identity:", recognition_result["predicted_identity"])

            print("Cosine similarity:", recognition_result["similarity"])

            print("Recognition result:", recognition_result["status"])

        entry.update(
            {
                "shqs_score": round(float(score), 2),
                "quality_factors": {k: float(v) for k, v in factors.items()},
                "decision": decision,
                "enhancement_method": method,
                "status": status,
                "output_image": str(destination),
                "recognition": recognition_result,
            }
        )

        report.append(entry)

        print("Enhancement:", method)
        print("Status:", status)

    # STAGE 4: GENERATE REPORT

    final_report = {
        "input_image": str(image_path),
        "total_faces": len(faces),
        "accepted_faces": sum(f["accepted"] for f in faces),
        "faces": report,
    }

    report_path = output / "report.json"

    with open(report_path, "w") as file:
        json.dump(final_report, file, indent=4)

    print("\n===== PIPELINE SUMMARY =====")
    print("Total faces:", len(faces))
    print("NO:", sum(r.get("decision") == "NO" for r in report))
    print("SOFT:", sum(r.get("decision") == "SOFT" for r in report))
    print("STRONG:", sum(r.get("decision") == "STRONG" for r in report))

    print("\nReport saved:", report_path)

    return final_report


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python Integration/pipeline.py images/test1.jpg")
        sys.exit(1)

    run_pipeline(sys.argv[1])
