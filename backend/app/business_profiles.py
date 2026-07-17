from __future__ import annotations

from dataclasses import dataclass
from re import sub
from typing import Literal


BusinessType = Literal["dental", "salon", "auto_repair"]


@dataclass(frozen=True)
class ServiceCatalogItem:
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class BusinessProfile:
    business_type: BusinessType
    label: str
    example_business_name: str
    services: tuple[ServiceCatalogItem, ...]


BUSINESS_NAME_MAX_LENGTH = 80
DEFAULT_BUSINESS_TYPE: BusinessType = "dental"


BUSINESS_PROFILES: dict[BusinessType, BusinessProfile] = {
    "dental": BusinessProfile(
        business_type="dental",
        label="Dental Clinic",
        example_business_name="BrightSmile Dental",
        services=(
            ServiceCatalogItem("dental cleaning", ("cleaning", "teeth cleaning")),
            ServiceCatalogItem("dental exam", ("checkup", "check-up", "exam")),
            ServiceCatalogItem("teeth whitening", ("whitening",)),
            ServiceCatalogItem("filling", ("cavity filling", "dental filling")),
            ServiceCatalogItem(
                "emergency dental visit",
                ("emergency visit", "tooth pain visit"),
            ),
        ),
    ),
    "salon": BusinessProfile(
        business_type="salon",
        label="Salon",
        example_business_name="Luxe Hair Studio",
        services=(
            ServiceCatalogItem("haircut", ("hair cut", "trim")),
            ServiceCatalogItem("blowout", ("blow dry", "blow-dry")),
            ServiceCatalogItem("hair color", ("color", "root touch up", "root touch-up")),
            ServiceCatalogItem("manicure", ("nails", "basic manicure")),
            ServiceCatalogItem("facial", ("skin facial",)),
        ),
    ),
    "auto_repair": BusinessProfile(
        business_type="auto_repair",
        label="Auto Repair Shop",
        example_business_name="TurboFix Garage",
        services=(
            ServiceCatalogItem("oil change", ("oil service",)),
            ServiceCatalogItem("brake inspection", ("brake check", "brakes")),
            ServiceCatalogItem("tire rotation", ("rotate tires", "tyre rotation")),
            ServiceCatalogItem("battery diagnostic", ("battery check",)),
            ServiceCatalogItem("engine diagnostic", ("check engine light", "diagnostic")),
        ),
    ),
}


def get_business_profile(business_type: BusinessType) -> BusinessProfile:
    try:
        return BUSINESS_PROFILES[business_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported business type: {business_type}") from exc


def default_business_profile() -> BusinessProfile:
    return get_business_profile(DEFAULT_BUSINESS_TYPE)


def is_supported_business_type(value: str) -> bool:
    return value in BUSINESS_PROFILES


def sanitize_business_name(
    business_name: str,
) -> str:
    cleaned_name = sub(r"[\x00-\x1F\x7F]", " ", business_name)
    cleaned_name = sub(r"\s+", " ", cleaned_name).strip()

    if not cleaned_name:
        raise ValueError("business_name is required")

    return cleaned_name[:BUSINESS_NAME_MAX_LENGTH]


def supported_service_names(profile: BusinessProfile) -> list[str]:
    return [service.name for service in profile.services]


def match_supported_service(
    profile: BusinessProfile,
    requested_service: str,
) -> str | None:
    requested = _normalize_service(requested_service)
    if not requested:
        return None

    for service in profile.services:
        candidates = (service.name, *service.aliases)
        normalized_candidates = [_normalize_service(candidate) for candidate in candidates]
        if any(
            requested == candidate
            or requested in candidate
            or candidate in requested
            for candidate in normalized_candidates
        ):
            return service.name

    return None


def _normalize_service(value: str) -> str:
    cleaned = sub(r"[^a-z0-9]+", " ", value.lower())
    return sub(r"\s+", " ", cleaned).strip()
