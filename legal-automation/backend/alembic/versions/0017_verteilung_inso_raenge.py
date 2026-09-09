"""Verteilung: § 39-Unterraenge, § 52-Ausfall, § 189-Rueckstellung, §§ 53-55

Der Verteilungsrechner behandelte alle nachrangigen Forderungen als einen Topf
und kannte weder das Ausfallprinzip fuer Absonderungsberechtigte noch die
Rueckstellung fuer bestrittene Forderungen. Die Masseverbindlichkeiten musste
der Aufrufer selbst herausrechnen.

Revision ID: 0017
Revises: 0016
"""
import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # § 52 InsO: Absonderungsberechtigte nehmen nur mit ihrem Ausfall teil
    op.add_column(
        "insolvency_claims",
        sa.Column("secured_recovery", sa.Numeric(14, 2), nullable=False, server_default="0"),
    )

    # §§ 53-55 InsO vorweg; § 189 InsO Rueckstellung
    op.add_column(
        "distributions",
        sa.Column("mass_liabilities", sa.Numeric(14, 2), nullable=False, server_default="0"),
    )
    op.add_column(
        "distributions",
        sa.Column("withheld_sum", sa.Numeric(14, 2), nullable=False, server_default="0"),
    )

    op.add_column(
        "distribution_items",
        sa.Column("participating_amount", sa.Numeric(14, 2), nullable=False, server_default="0"),
    )
    op.add_column(
        "distribution_items",
        sa.Column("rank", sa.String(30), nullable=False, server_default="insolvenz_38"),
    )
    op.add_column(
        "distribution_items",
        sa.Column("withheld", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Bestandsdaten: ohne Sicherheitenverwertung entspricht der teilnehmende
    # Betrag dem festgestellten.
    op.execute("UPDATE distribution_items SET participating_amount = established_amount")


def downgrade() -> None:
    op.drop_column("distribution_items", "withheld")
    op.drop_column("distribution_items", "rank")
    op.drop_column("distribution_items", "participating_amount")
    op.drop_column("distributions", "withheld_sum")
    op.drop_column("distributions", "mass_liabilities")
    op.drop_column("insolvency_claims", "secured_recovery")
