import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
CUTOFF_YEAR = 2025
MIN_OBSERVATION_DAYS = 300

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 기온으로 연평균기온을 계산하고, 선형 회귀로 선택한 연도의 "
    "예상 기온을 살펴봅니다."
)
st.caption("학습 기준: 2025년까지의 자료 중 유효한 관측일이 300일 이상인 연도")

@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중입니다...")
def load_yearly_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["날짜", "평균기온"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "CSV에서 필요한 열을 찾을 수 없습니다: " + ", ".join(missing)
        )

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 자료만 사용하고, 날짜 기준으로 관측일 수를 계산합니다.
    df = df[df["연도"] <= CUTOFF_YEAR]
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도는 학습 자료에서 제외합니다.
    yearly = yearly[yearly["관측일수"] >= MIN_OBSERVATION_DAYS]
    return yearly.sort_values("연도").reset_index(drop=True)

try:
    yearly = load_yearly_data()
except Exception as exc:
    st.error("데이터를 불러오거나 처리하지 못했습니다.")
    st.exception(exc)
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 수 있을 만큼 충분한 연도 데이터가 없습니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 1차 다항식 회귀: 예상 기온 = 기울기 × 연도 + 절편
slope, intercept = np.polyfit(years, temps, 1)
correlation = float(np.corrcoef(years, temps)[0, 1])
start_year = int(years.min())
end_year = int(years.max())
training_count = len(yearly)

# 최근 20년: 학습 자료에서 가장 마지막으로 남은 연도를 기준으로 20년 구간을 정합니다.
recent_end_year = end_year
recent_start_year = recent_end_year - 19
recent = yearly[yearly["연도"].between(recent_start_year, recent_end_year)].copy()
if len(recent) >= 2:
    recent_slope, recent_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    recent_start_used = int(recent["연도"].min())
    recent_end_used = int(recent["연도"].max())
else:
    recent_slope, recent_intercept = np.nan, np.nan
    recent_start_used = recent_end_used = None

selected_year = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)
predicted_temp = slope * selected_year + intercept

st.subheader(f"{selected_year}년 예상 연평균기온")
st.markdown(
    f"<div style='font-size:3rem;font-weight:700;line-height:1.2;'>"
    f"{predicted_temp:.2f} °C</div>",
    unsafe_allow_html=True,
)
st.caption("과거 연도별 평균기온에 맞춘 선형 회귀 직선으로 계산한 추정값입니다.")

st.subheader("100년에 기온이 얼마나 오를까?")
trend_all, trend_recent = st.columns(2)
with trend_all:
    st.metric("전체 학습 기간의 변화", f"{slope * 100:+.2f} °C / 100년")
    st.caption(
        f"{start_year}~{end_year}년 · {training_count}개 연도 사용"
    )
with trend_recent:
    if np.isfinite(recent_slope):
        st.metric("최근 20년의 변화", f"{recent_slope * 100:+.2f} °C / 100년")
        st.caption(
            f"최근 20년 구간 중 유효 자료 {recent_start_used}~{recent_end_used}년, "
            f"{len(recent)}개 연도 사용"
        )
    else:
        st.metric("최근 20년의 변화", "계산할 수 없음")
        st.caption("최근 20년 구간에 유효한 연도가 2개 미만입니다.")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("회귀 학습 연도 수", f"{training_count:,}년")
metric2.metric("시작 연도", f"{start_year}년")
metric3.metric("끝 연도", f"{end_year}년")
metric4.metric("상관계수", f"{correlation:.3f}")

st.subheader("연도별 평균기온과 회귀 직선")
line_years = np.linspace(start_year, 2100, 300)
line_temps = slope * line_years + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>연평균기온: %{y:.2f} °C"
            "<br>관측일수: %{customdata[0]}일<extra></extra>"
        ),
    )
)
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate="연도: %{x:.0f}년<br>회귀 예상: %{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상",
        marker=dict(size=13, symbol="star"),
        hovertemplate="%{x}년 예상 기온: %{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="범례",
    margin=dict(l=20, r=20, t=25, b=20),
)
fig.update_xaxes(range=[1900, 2100])
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 회귀 직선 정보")
st.write(
    f"- 학습에 사용한 연도 수: **{training_count}개**\n"
    f"- 시작 연도: **{start_year}년**\n"
    f"- 끝 연도: **{end_year}년**\n"
    f"- 전체 기간 회귀식: **예상 기온 = {slope:.5f} × 연도 + ({intercept:.2f})**\n"
    + (
        f"- 최근 20년 회귀식: **예상 기온 = {recent_slope:.5f} × 연도 + ({recent_intercept:.2f})**"
        if np.isfinite(recent_slope)
        else "- 최근 20년 회귀식: 계산할 수 없음"
    )
)
st.write(
    f"- 상관계수(연도와 연평균기온): **{correlation:.3f}**"
)

with st.expander("학습에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(
            columns={
                "연도": "연도",
                "연평균기온": "연평균기온(°C)",
                "관측일수": "관측일수(일)",
            }
        ).round({"연평균기온": 2}),
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "주의: 이 결과는 연도와 기온 사이의 직선 관계를 과거 자료에 맞춘 단순 예측입니다. "
    "장기 기후 예측 모델이나 공식 기상 전망은 아닙니다."
)
