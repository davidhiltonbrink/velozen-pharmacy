"""
Velozen AI — SQLAlchemy ORM models.

Tables
------
  pharmacies          - pilot pharmacy registry (built for multi-pharmacy from day one)
  drugs               - SKU catalog (NDC codes, names, categories)
  dispensing_records  - weekly fill counts per drug per pharmacy
  inventory_snapshots - weekly on-hand inventory levels
  forecasts           - model-generated demand predictions
  model_eval_metrics  - per-SKU accuracy metrics from each training run
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, ForeignKey,
    Integer, Numeric, String, func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Pharmacy(Base):
    __tablename__ = "pharmacies"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    name         = Column(String(200), nullable=False)
    city         = Column(String(100))
    state        = Column(String(2))
    zip_code     = Column(String(10))
    pms_platform = Column(String(50))   # PioneerRx | QS1 | Liberty | other
    created_at   = Column(DateTime, server_default=func.now())

    dispensing_records  = relationship("DispensingRecord",  back_populates="pharmacy")
    inventory_snapshots = relationship("InventorySnapshot", back_populates="pharmacy")
    forecasts           = relationship("Forecast",          back_populates="pharmacy")


class Drug(Base):
    __tablename__ = "drugs"

    ndc               = Column(String(11), primary_key=True)
    drug_name         = Column(String(200), nullable=False)
    category          = Column(String(50),  nullable=False)   # chronic | seasonal | other
    avg_daily_rx_fills = Column(Numeric(8, 2))
    pack_sizes        = Column(String(50))                    # e.g. "30|90"
    unit_cost_usd     = Column(Numeric(10, 2))
    shelf_life_days   = Column(Integer)
    created_at        = Column(DateTime, server_default=func.now())

    dispensing_records  = relationship("DispensingRecord",  back_populates="drug")
    inventory_snapshots = relationship("InventorySnapshot", back_populates="drug")
    forecasts           = relationship("Forecast",          back_populates="drug")
    eval_metrics        = relationship("ModelEvalMetric",   back_populates="drug")


class DispensingRecord(Base):
    __tablename__ = "dispensing_records"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    pharmacy_id  = Column(Integer, ForeignKey("pharmacies.id"), nullable=True)
    ndc          = Column(String(11), ForeignKey("drugs.ndc"),  nullable=False)
    dispense_date = Column(Date, nullable=False)
    quantity     = Column(Numeric(10, 2), nullable=False)
    source       = Column(String(50))   # synthetic | synthea | cms_medicaid | pms_pionerrx
    created_at   = Column(DateTime, server_default=func.now())

    pharmacy = relationship("Pharmacy", back_populates="dispensing_records")
    drug     = relationship("Drug",     back_populates="dispensing_records")


class InventorySnapshot(Base):
    __tablename__ = "inventory_snapshots"

    id                  = Column(BigInteger, primary_key=True, autoincrement=True)
    pharmacy_id         = Column(Integer, ForeignKey("pharmacies.id"), nullable=True)
    ndc                 = Column(String(11), ForeignKey("drugs.ndc"),  nullable=False)
    snapshot_date       = Column(Date, nullable=False)
    on_hand_units       = Column(Numeric(10, 2))
    dispensed_past_week = Column(Numeric(10, 2))
    reorder_qty         = Column(Numeric(10, 2))
    expired_units       = Column(Numeric(10, 2))
    stockout_flag       = Column(Boolean, default=False)
    expiry_date         = Column(Date)
    inventory_value_usd = Column(Numeric(12, 2))
    created_at          = Column(DateTime, server_default=func.now())

    pharmacy = relationship("Pharmacy", back_populates="inventory_snapshots")
    drug     = relationship("Drug",     back_populates="inventory_snapshots")


class Forecast(Base):
    __tablename__ = "forecasts"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    pharmacy_id   = Column(Integer, ForeignKey("pharmacies.id"), nullable=True)
    ndc           = Column(String(11), ForeignKey("drugs.ndc"),  nullable=False)
    drug_name     = Column(String(200))
    category      = Column(String(50))
    forecast_week = Column(Date, nullable=False)
    y             = Column(Numeric(10, 2))     # actual fill count (when available)
    yhat          = Column(Numeric(10, 2), nullable=False)
    model_version = Column(String(50))
    generated_at  = Column(DateTime, server_default=func.now())

    pharmacy = relationship("Pharmacy", back_populates="forecasts")
    drug     = relationship("Drug",     back_populates="forecasts")


class ModelEvalMetric(Base):
    __tablename__ = "model_eval_metrics"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    model_version  = Column(String(50))
    ndc            = Column(String(11), ForeignKey("drugs.ndc"), nullable=False)
    drug_name      = Column(String(200))
    category       = Column(String(50))
    test_weeks     = Column(Integer)
    mae            = Column(Numeric(10, 4))
    rmse           = Column(Numeric(10, 4))
    mape_pct       = Column(Numeric(8,  4))
    bias           = Column(Numeric(10, 4))
    high_error_flag = Column(Boolean, default=False)
    evaluated_at   = Column(DateTime, server_default=func.now())

    drug = relationship("Drug", back_populates="eval_metrics")
